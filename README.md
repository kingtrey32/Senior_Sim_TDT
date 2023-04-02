# Senior_Sim_TDT
Senior capstone project for Spring 2023 CS4900

## Project information:
The project performs the following steps:
	
1.	Generates and android app that will be uploaded to a virtual machine that can identify clothing
2.	Using the FashionMnist data set as its Data set 
3.	Allowing the user to select a 20x20 pixel photo from the virtual phones gallery 

Requirements:

•	Android Studio: https://developer.android.com/studio
•	Anaconda: https://www.anaconda.com/products/distribution
•	Spyder: Installed through the Anaconda Enviorment
•	Pytorch: https://pytorch.org/get-started/locally/
•	(If GPU use `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`)
•	(If CPU use `conda install pytorch torchvision torchaudio cpuonly -c pytorch`)
•	FashionMnist Dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist
•	Dr.Ahana Roy’s Model to do the identifying ( supplied In the GIthub)

Execution Instructions:

1.	Install all the above programs and tools to your local machine ( make a new folder and make sure there all located in 		that folder for file pathing reasons)
2.	Open the conda Powershell prompt and use one of the above commands based on your pc’s hardware
3.	Our program does not have the capability of putting the trained model on the virtual phone ( this feature wil be updated 	 later)
4.	Open up spyder notice it creates a temp file do not exit that file 
5.	Make a copy of the repository on github and extract in the same folder where all the programs are stored
6.	Find the model_creator file and copy and paste it into your temp file
7.	Open a new file and copy and past the model code into it and run it so it saves to the program
8.	Once copied run the program( notice we did not mention anything about the FashionMnist data set that’s because our 		program downloads it from the internet and store it in the prper location for you)
