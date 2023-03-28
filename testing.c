#******testing the model**************************************************************************
        if not (num_epochs % 50): #the same as "if num_wpochs % 50 == 0"
            total = 0
            correct = 0
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                
                test = Variable(images.view(100, 1, 28, 28))
                
                outputs = model(test)
                
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(num_epochs)
            #accuracy_list.append(accuracy)
        
        if not (num_epochs % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%" .format(num_epochs, loss.data, accuracy))
