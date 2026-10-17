# Releasing

A release is one version number shared by three things:

- **`factorion-server`** on PyPI: the wheels players run with `uvx factorion-server@X.Y.Z`
  (macOS arm64 and Linux x86_64).
- **`factorion`** on the [Factorio mod portal](https://mods.factorio.com/mod/factorion),
  whose chat help names the matching `uvx factorion-server@X.Y.Z`.
- **A checkpoint**: the mod's default (`checkpoint` in
  `factorion-mod/mod/inference.lua`), attached to the GitHub release as `<run_id>.pt`,
  where the server finds it without W&B.

## Cutting one

Push a tag, or run the **release** workflow (Actions → release → Run workflow) with a
version, which tags the chosen commit for you:

```bash
git tag v0.7.0 && git push origin v0.7.0
```

`.github/workflows/release.yml` stamps the version into `pyproject.toml` and the
mod's `info.json` (both read `0.0.0` in the repo), builds the wheels, fetches the
checkpoint from W&B and generates a factory with it on the installed wheel, then
publishes the GitHub release, PyPI and the mod portal. A checkpoint the release's
code can't load fails the release before anything is published, so after an
architecture change, train a new SFT run and point `inference.lua` at it first.

## Versions

While we're on 0.x:

- **0.MINOR.0**: code changes that break something, e.g. the model architecture
  (old checkpoints stop loading) or the mod ↔ server protocol.
- **0.MINOR.PATCH**: anything backwards compatible, usually a better checkpoint.
- **1.0.0**: the first release we'd call stable.

## One-time setup

- **PyPI**: add a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
  for project `factorion-server`, owner `beyarkay`, repository `factorion`, workflow
  `release.yml`, environment `pypi`.
- **Mod portal**: create an API key on your [factorio.com profile](https://factorio.com/profile)
  with the *ModPortal: Publish Mods* and *ModPortal: Upload Mods* permissions, and save it
  as the repository secret `FACTORIO_API_KEY`. The first release creates the mod's page.
- **W&B**: the existing `WANDB_API_KEY` secret downloads the checkpoint.
