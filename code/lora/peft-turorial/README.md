
## Training
Training without lora
```bash
python train.py
```

Training with lora
```bash
python train.py --with_lora
```

## Testing
Testing without lora
```bash
python eval.py
```
outputs:
```bash
Rogue1: 42.771837%
rouge2: 14.955854%
rougeL: 32.591311%
rougeLsum: 32.500095%
```

with lora
```bash
python eval.py --with_lora
```
Outputs:
```bash
Rogue1: 46.917517%
rouge2: 21.502901%
rougeL: 37.822022%
rougeLsum: 38.091122%
```
