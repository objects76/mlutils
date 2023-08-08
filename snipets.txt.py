
    checkpoint_file = "unet_checkpoint.pth.tar"
    
    epoch_start = 0
    def checkpoint_save(epoch, scores):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scores': scores,
                    }, checkpoint_file)
        print(f'checkpoint saved: epoch={epoch}')

    def checkpoint_load():
        print(checkpoint_file)
        if not os.path.exists(checkpoint_file):
            return
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        scores = checkpoint['scores']
        print(f'checkpoint loaded: epoch={epoch}, \n\t score(acc/iou/dice)={scores}')
        print()
        return epoch+1 # next epoch have to handled.



    epoch_secs = MovingAverage(window_size=3)

    for epoch in range(epoch_start,NUM_EPOCHS):

        remained_sec = int(epoch_secs.avg*(NUM_EPOCHS-epoch-1))
        print(f"{epoch+1}/{NUM_EPOCHS}: estimated remained time: {datetime.timedelta(seconds=remained_sec)} sec")
        secs = timeit.timeit(lambda: train_fn(train_loader, model, optimizer, loss_fn, scaler), number=1 )
        epoch_secs.update(secs)
