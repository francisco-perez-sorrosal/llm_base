# llm_foundation


## Build

```
pixi install
```

## Test

```
pixi r pytest
```

### Development

Make changes and push:
```
# Increment version
pixi project version patch
pixi install
# Push to github
git add .
git commit -m "xxxx < Intentional >
git push origin main
```

## Extras

Create virtual env (Only if poetry is  used)

```sh
pyenv virtualenv 3.11.1 llm_foundation_env
cd ~/dev/llm_foundation
pyenv local llm_foundation_env
```

Add pixi dependencies:

```sh
pixi add crewai
```
