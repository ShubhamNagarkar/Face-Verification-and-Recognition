# Face-Verification-and-Recognition

<h3> Face Verification </h3>
<br>
1) "Is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person      carrying the passport) are the correct person. <br>
2) A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
<br>
To address this problem, the system uses pre-trained Inception network to encode 96x96 RGB images to 128-dimensional vector to perform 1:1 Face Verification.
<br>
<h3>System Architecture:</h3>

![overview diagram](/architecture.png)

<h3> Face Recognition </h3>
<br>
1) In case of Face Recognition, the model takes as input an image, and figures out if it is one of the authorized persons.
<br>
2) This is a 1:K matching problem as every input image is compared to all the images in the database.
<br>
<h3> Basic Model: </h3>

![Architecture](/Inception.png)

Given an input image, the Inception Network computes a 128-dimensional encoding. This encoding is then compared with the encodings in the database.
