# delta_src
Script directory

## model

### dynamic_snn
SNN that dynamically changes the time constant and membrane resistance.

If you import it, you can use it like a normal pytorch model.

## generate dataset
Generate data for learning
Scripts are separated for each dataset


## gen cache
Generate cache data for learning
If you do this, the time to generate batches from the dataloader is omitted, making it faster