# ML-Docker-Jenkins
This project is based on the integration of Machine Learning with DevOps.

This integration is the solution to the problems of deployment of the Machine Learning and Deep Learning models.
Most of the ML and DL models are not deployed, due to the maintenance and ability to increase the accuracy of the model when deployed, the manual tweaking of the code, the manual changes to be made  in the hyperparameters, etc.

This integration of Machine Learning with the DevOps i.e., the Automation is the solution for such problems. It will help in increasing the accuracy of the model, handling the tweaking of the code by itself, so no human efforts are needed after setting up the system just once.

*Docker containers and Jenkins will do the Job!*

Here is the detailed description of the whole task:

1. Create a Docker image that has python and all the ML and DL packages installed using Dockerfile. 

2. When we launch this image, it should automatically start to train the ML model in the container.

3. Create a job chain of job1, job2, job3, job4 and job5 using build pipeline plugin in Jenkins 

4. Job1: Pull the Github repo automatically when some developers make changes in the repo.

5. Job2: By looking at the code or program file, Jenkins should automatically start the respective machine learning software installed interpreter to deploy code and start training( eg. If code uses CNN, then Jenkins should start the container that has already installed all the softwares required for the CNN processing).

6. Job3: Train your model and predict accuracy or metrics.

7. Job4: if metrics accuracy is less than 80%, then tweak the machine learning model architecture to get better accuracy.

8. Job5: Retrain the model or notify that the best model is being created.

9. Create One extra job, job6 to monitor: If the container where the model is running, fails due to any reason then this job should automatically restart the container from the last point.
