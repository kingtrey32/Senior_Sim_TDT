#****training the model**********************************************************************
    for images, labels in train_loader:
        #transferring images and laels to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        #forward pass
        outputs = model(train)
        loss = error_rate(outputs, labels)
        
        #initializing gradient at 0 for each batch
        optimizer.zero_grad()
        
        #backpropogation of error found
        loss.backward()
        
        #optimizing parameters given loss rate
        optimizer.step()
        
        num_epochs += 1
