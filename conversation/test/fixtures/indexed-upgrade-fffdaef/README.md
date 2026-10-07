# Reproducing the historical indexed fixture

This fixture preserves the bytes emitted by commit
`fffdaef7f2be1d50461a8891c3c0e29706dfb443`. The capture manifest records the
encoder hash, generator hash, output hash and dependency version. Install the
Llumiverse workspace dependencies first; the capture requires Zod 4.6.5 and
Node support for `registerHooks` and `--experimental-transform-types`.

From the Llumiverse checkout root, run:

```sh
capture_repo="$(pwd)"
capture_dir="$(mktemp -d)"
mkdir -p "$capture_dir/fixture" "$capture_dir/historical"
git archive fffdaef7f2be1d50461a8891c3c0e29706dfb443 conversation/src |
    tar -x -C "$capture_dir/historical"
cp conversation/test/fixtures/indexed-upgrade-fffdaef/capture.mjs "$capture_dir/fixture/"
cp conversation/test/fixtures/indexed-upgrade-fffdaef/provenance.json "$capture_dir/fixture/"
node --experimental-transform-types "$capture_dir/fixture/capture.mjs" "$capture_repo"
cmp conversation/test/fixtures/indexed-upgrade-fffdaef/historical-fffdaef-artifacts.json \
    "$capture_dir/fixture/historical-fffdaef-artifacts.json"
```

The explicit checkout argument supplies installed dependencies. The provenance
retains the original capture location as historical metadata, so generated
artifact bytes remain identical across checkout locations. Capture writes only
to the temporary fixture directory.
