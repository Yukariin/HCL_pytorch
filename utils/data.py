def get_data_generator(dataloader, start_epoch=0):
    ep = start_epoch
    while True:
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(ep)
        for batch in dataloader:
            yield batch
        ep += 1
