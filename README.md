# Face-recognition-based-attendance-system

#How it works:

This is a face recognition based attendance system built using OpenCV , tensorflow and keras. It consists of two python files , one named as att_register and another as att_scan.
The register file works by first detecting a face using haar cascade and then sending the detected face to a deep learning model(vgg_face model) to create encodings for the particular
face. It also asks for name for identification and stores the encoding and the name in a CSV file. Similarly the scan file scans for the face and detects the face and send the face to the
vgg_face model and calculates the encoding.Then this encoding is compared with the registere encodings in the CSV file and compares using COSINE similarity . The encoding with least 
cosine angle is the identity for the new scanned face. Then this new scanned face/name will be stored in another CSV with date and time.

#Running the file:

First run the att_register.py file and provide the path to a csv file . Then scan your face and provide your name and provide your choice to register more faces. Then run the att_scan.py
file and provide the directory where you stored the registration csv and provide a new directory to where the attendance csv to be stored. Then thats all your face recognition based attendance system is done.
