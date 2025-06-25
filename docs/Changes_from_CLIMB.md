# Changes

- No need for manual multi-gpu thingy, huggingface trainer supports that out of the box now, so this part of the code is removed:
```
If you are extending Hugging Face’s Trainer class, then Accelerate is already integrated. This means:

It handles mixed precision (fp16=True or bf16=True)
It moves your model/optimizer/dataloaders to GPU(s)
It supports multi-GPU and TPU automatically
```

