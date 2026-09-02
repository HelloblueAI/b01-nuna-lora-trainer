Identity seed is regenerated from private B01.beta:

```bash
cd ../B01.beta
pnpm run training:export-lora-dataset -- --out ../b01-nuna-lora-trainer/datasets/identity-seed.json
```

`from-b01.json` and `live-*.json` are gitignored. Never commit user conversations.
