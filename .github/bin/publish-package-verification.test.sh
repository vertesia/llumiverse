#!/usr/bin/env bash
# Real Git/tar fixtures exercise both publisher modes; registry writes are substituted.
set -euo pipefail
if [[ "${1:-}" != "--isolated-fixture" ]]; then
  exec env -i PATH="$PATH" HOME="$HOME" bash "${BASH_SOURCE[0]}" --isolated-fixture "$@"
fi
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin"
cat > "$work/bin/npm" <<'JS'
#!/usr/bin/env node
const fs=require('node:fs'),p=JSON.parse(fs.readFileSync('package.json'));
const args=process.argv.slice(2);
if(args[0]==='view'){console.log('{"error":{"code":"E404"}}');process.exit(1);}
if(args[0]==='version'){p.version=args[1];fs.writeFileSync('package.json',JSON.stringify(p)+'\n');}
else if(args[0]==='pkg'&&args[1]==='get')console.log(JSON.stringify(p[args[2]]));
else if(args[0]==='pkg'&&args[1]==='set'){
 const index=args[2].indexOf('='),key=args[2].slice(0,index),value=args[2].slice(index+1);
 p[key]=args.includes('--json')?JSON.parse(value):value;fs.writeFileSync('package.json',JSON.stringify(p)+'\n');
}
else throw new Error('Unexpected npm call '+args.join(' '));
JS
cat > "$work/bin/pnpm" <<'JS'
#!/usr/bin/env node
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const args=process.argv.slice(2),log=process.env.FIXTURE_LOG;
if(args[0]==='build')process.exit(0);
if(args[0]==='-r'){
 for(const pkg of ['common','core','drivers'])cp.execFileSync('npm',args.slice(args.indexOf('exec')+2),{cwd:pkg,stdio:'inherit'});
 process.exit(0);
}
if(args[0]==='pack'){
 const p=JSON.parse(fs.readFileSync('package.json'));
 for(const [name] of Object.entries(p.dependencies||{}))if(name.startsWith('@llumiverse/'))p.dependencies[name]=p.version;
 const destination=args[args.indexOf('--pack-destination')+1],staging=path.join(destination,'staging');
 fs.mkdirSync(path.join(staging,'package/lib'),{recursive:true});
 fs.writeFileSync(path.join(staging,'package/package.json'),JSON.stringify(p));
 fs.writeFileSync(path.join(staging,'package/lib/index.d.ts'),'fixture');
 if(p.name!==process.env.BAD_PACKAGE)fs.writeFileSync(path.join(staging,'package/lib/index.js'),'fixture');
 const archive=path.join(destination,'package.tgz');
 cp.execFileSync('tar',['-czf',archive,'-C',staging,'package']);
 const hash=crypto.createHash('sha256').update(fs.readFileSync(archive)).digest('hex');
 fs.appendFileSync(log,`pack:${p.name}:${hash}\n`);process.exit(0);
}
if(args[0]==='publish'){
 if(!args[1]?.endsWith('.tgz'))throw new Error('Publisher must use the verified archive');
 const p=JSON.parse(cp.execFileSync('tar',['-xzOf',args[1],'package/package.json'],{encoding:'utf8'}));
 const hash=crypto.createHash('sha256').update(fs.readFileSync(args[1])).digest('hex');
 const events=fs.readFileSync(log,'utf8').trim().split('\n');
 if(events.filter(line=>line.startsWith('pack:')).length!==3)throw new Error('Published before every package was packed');
 if(!events.includes(`pack:${p.name}:${hash}`))throw new Error('Published archive differs from verified archive');
 fs.appendFileSync(log,`publish:${p.name}:${args.includes('--dry-run')?'dry':'real'}\n`);process.exit(0);
}
throw new Error('Unexpected pnpm call '+args.join(' '));
JS
chmod +x "$work/bin/npm" "$work/bin/pnpm"
export PATH="$work/bin:$PATH"
for mode in true false; do
  for valid in true false; do
    source_dir="$work/source-$mode-$valid"
    git init -q -b main "$source_dir"
    git init -q --bare "$source_dir-remote"
    cd "$source_dir"
    git config user.name Test
    git config user.email test@example.test
    printf '{"name":"@llumiverse/parent","version":"1.6.0-dev"}\n' > package.json
    printf 'lockfileVersion: 9\n' > pnpm-lock.yaml
    for pkg in common core drivers; do
      mkdir "$pkg"
      printf '{"name":"@llumiverse/%s","version":"1.6.0-dev","main":"./lib/index.js","types":"./lib/index.d.ts"}\n' "$pkg" > "$pkg/package.json"
    done
    git add .
    git commit -qm fixture
    before="$(git rev-parse HEAD)"
    git remote add origin "$source_dir-remote"
    git push -q origin main
    export FIXTURE_LOG="$work/events-$mode-$valid"
    : > "$FIXTURE_LOG"
    export BAD_PACKAGE=''
    if [[ "$valid" == false ]]; then export BAD_PACKAGE=@llumiverse/drivers; fi
    result=0
    bash "$script_dir/publish-all-packages.sh" --ref main --release-type snapshot --bump-type keep --dry-run "$mode" > "$work/run-$mode-$valid.log" 2>&1 || result=$?
    if [[ "$valid" == true ]]; then
      [[ "$result" == 0 ]] || { cat "$work/run-$mode-$valid.log"; exit 1; }
      [[ "$(grep -c '^publish:' "$FIXTURE_LOG")" == 3 ]]
      if [[ "$mode" == true ]]; then grep -q ':dry$' "$FIXTURE_LOG"; else grep -q ':real$' "$FIXTURE_LOG"; fi
    else
      [[ "$result" != 0 ]]
      ! grep -q '^publish:' "$FIXTURE_LOG"
      grep -q 'declared target ./lib/index.js is missing' "$work/run-$mode-$valid.log"
    fi
    after="$(git --git-dir="$source_dir-remote" rev-parse main)"
    if [[ "$valid" == false || "$mode" == true ]]; then [[ "$before" == "$after" ]]; else [[ "$before" != "$after" ]]; fi
  done
done
echo 'Real/dry publication verifies every archive before any publish; invalid packages block registry and Git writes'
