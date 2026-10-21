#!/usr/bin/env bash
# Exercise the real publisher with real Git and a local npm registry substitute.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin" "$work/registry"
export REGISTRY_FIXTURE="$work/registry"
cat > "$work/bin/npm" <<'JS'
#!/usr/bin/env node
const fs=require('node:fs'),path=require('node:path');
const args=process.argv.slice(2),file=path.resolve('package.json');
if(args[0]==='view') {
 const spec=args[1], data=path.join(process.env.REGISTRY_FIXTURE,encodeURIComponent(spec)+'.json');
 if(!fs.existsSync(data)){console.log('{"error":{"code":"E404"}}');process.exit(1);}
 console.log(fs.readFileSync(data,'utf8'));process.exit(0);
}
const p=JSON.parse(fs.readFileSync(file,'utf8'));
if(args[0]==='version')p.version=args[1];
else if(args[0]==='pkg' && args[1]==='get'){console.log(JSON.stringify(p[args[2]]));process.exit(0);}
else if(args[0]==='pkg' && args[1]==='set'){
 const index=args[2].indexOf('='),key=args[2].slice(0,index),value=args[2].slice(index+1);
 p[key]=args.includes('--json')?JSON.parse(value):value;
}else throw new Error('Unexpected npm call: '+args.join(' '));
fs.writeFileSync(file,JSON.stringify(p)+'\n');
JS
cat > "$work/bin/pnpm" <<'JS'
#!/usr/bin/env node
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const args=process.argv.slice(2);
if(args[0]==='build')process.exit(0);
if(args[0]==='-r'){
 const offset=args.indexOf('exec');
 for(const pkg of ['common','core','drivers']){
  const r=cp.spawnSync(args[offset+1],args.slice(offset+2),{cwd:path.resolve(pkg),stdio:'inherit'});
  if(r.status)process.exit(r.status);
 }process.exit(0);
}
if(args[0]==='publish'){
 const p=JSON.parse(fs.readFileSync('package.json','utf8')),dir=process.env.REGISTRY_FIXTURE;
 if(p.name===process.env.FAIL_PACKAGE && !fs.existsSync(path.join(dir,'failed-once'))){
  fs.writeFileSync(path.join(dir,'failed-once'),'');process.exit(1);
 }
 fs.writeFileSync(path.join(dir,encodeURIComponent(p.name+'@'+p.version)+'.json'),JSON.stringify(p));
 fs.appendFileSync(path.join(dir,'publishes'),p.name+'@'+p.version+'\n');process.exit(0);
}throw new Error('Unexpected pnpm call: '+args.join(' '));
JS
chmod +x "$work/bin/npm" "$work/bin/pnpm"
export PATH="$work/bin:$PATH"
git init -q -b main "$work/source"
git init -q --bare "$work/remote"
cd "$work/source"
git config user.name Test
git config user.email test@example.test
printf '{"name":"@llumiverse/parent","version":"1.6.0-dev"}\n' > package.json
printf 'lockfileVersion: 9\n' > pnpm-lock.yaml
for pkg in common core drivers; do
 mkdir "$pkg"; printf '{"name":"@llumiverse/%s","version":"1.6.0-dev"}\n' "$pkg" > "$pkg/package.json"
done
git add .
git commit -qm source
source_sha="$(git rev-parse HEAD)"
git remote add origin "$work/remote"
git push -q origin main
export RELEASE_SOURCE_SHA="$source_sha" VERSION_OVERRIDE=1.6.0-dev.20261021.123456Z
export COORDINATED_RELEASE=true FAIL_PACKAGE=@llumiverse/drivers
# A partial publish has a durable tag, and a retry must publish only the missing package.
if bash "$script_dir/publish-all-packages.sh" --ref main --release-type snapshot --bump-type keep --dry-run false; then
 echo 'Expected injected partial publish failure' >&2; exit 1
fi
tag_sha="$(git rev-parse "v$VERSION_OVERRIDE")"
[[ "$(git --git-dir="$work/remote" rev-parse "v$VERSION_OVERRIDE")" == "$tag_sha" ]]
# Main advances before the retry, after two immutable packages are already on the registry.
git checkout -q --detach "$source_sha"
printf 'new ordinary code\n' > newer.txt
git add newer.txt
git commit -qm newer
git push -q origin HEAD:refs/heads/main
git checkout -q --detach "$source_sha"
bash "$script_dir/publish-all-packages.sh" --ref main --release-type snapshot --bump-type keep --dry-run false
[[ "$(wc -l < "$REGISTRY_FIXTURE/publishes" | tr -d ' ')" == '3' ]]
[[ "$(git --git-dir="$work/remote" rev-parse "v$VERSION_OVERRIDE")" == "$tag_sha" ]]
[[ "$(git --git-dir="$work/remote" show main:newer.txt)" == 'new ordinary code' ]]
# A foreign existing version must fail before ANY new Git ref or registry write.
git checkout -q --detach "$source_sha"
export VERSION_OVERRIDE=1.6.0-dev.20261021.123457Z
printf '{"gitHead":"foreign"}\n' > "$REGISTRY_FIXTURE/%40llumiverse%2Fcore%401.6.0-dev.20261021.123457Z.json"
if bash "$script_dir/publish-all-packages.sh" --ref main --release-type snapshot --bump-type keep --dry-run false; then
 echo 'Expected foreign package preflight failure' >&2; exit 1
fi
! git --git-dir="$work/remote" rev-parse --verify "v$VERSION_OVERRIDE" 2>/dev/null
[[ "$(wc -l < "$REGISTRY_FIXTURE/publishes" | tr -d ' ')" == '3' ]]
echo 'Actual publisher: partial npm retry after main advances, stable tag identity, and preflight-before-writes passed'
