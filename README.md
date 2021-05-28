# ConvexHull-Object-Detection

# Usage

All source code is in the Code folder. All of our code is in the form of jupyter notebooks.

Preproceesing.ipynb : This file contains the pre-processing i.e. how to convert image data into convex hull.

Training.ipynb : This file contains all the models.

Post_proceesing.ipynb : This file contains how to convert output of our models into convex hull.

# Abstact

Abstract—We have multiple efficient methods for detecting objects within rectangular bounding boxes, but these rectangular boxes have large portions of background information. This leads to noise in the image. Also, for some cases like circular objects, oval-shaped objects, having a polygon encompassing their shape, may lead to 3-d interpretation of the object. With this view in mind, we have introduced the first Convex-hull object detector, based on Residual-Dense USegNet. Our architecture can detect objects of any arbitrary shape and place it inside the object’s convex hull. We have done experiments on baseline models and also on various variants of USegNet, but our model performs the best. The experiments are performed on the entire cityscapes- dataset, with the semantic segmentation as the ground truth.

# Dataset Prepration or Preprocessing :

We have taken cityscapes-dataset comprising real-life im- ages as well as semantic understanding of urban street scenes. The semantic segmentation consists of fine annotations of about 50 cities. The ground truth is provided by taking a semantic segmentation and finding the corresponding pixels with our object. We have taken 5 objects into focus and the part where the object is present is assigned as 1, others are 0 (background). After getting semantic segmentation of each object in one channel, we applied a convex hull algorithm on that channel and got a convex hull shape on that object. After getting the convex hull around our object, we further assigned the pixels which were a part of the background but a part of the convex hull as 1. So, we got a convex hull of objects as ground truth for training and real-life images as input. So, the shape of ground truth image is 128*256*5 (where 5 are no of channels and each channel is corresponding to 1 object) and shape of input image is 128*256*3. Original size of the image was 1024 x 2048, it’s size was reduced for faster computation

# Architecture

We have used USegNet as our base model, which is a combination of U-Net and Segnet architecture. We have added residual connection within a block and residual-dense skip connections across a block as a modification for better results.Residual and dense skip connections in the network enable short paths to be built directly from the output to each layer, alleviating the vanishing-gradient problem of a very deep network.They also add multiscale information along with it, also help us in incorporating both coarser and finer information. We call it Residual dense USegnet. USegNet is a hybrid architecture of U-Net and USegnet. The architecture is a U-shaped model, having different layers of convolutional, relu, batch normalization and pooling layers, as shown in the fig. 2. The initial feature map size is 128 x 256 x 3 i.e. the feature map has 3 channels, and the input is passed through 2 layers (1 layer consists of a convolution layer with kernel size =3, batch-normalization and RELU activation). The input of the first layer of one block is passed as an input to the first layer of the next block through residual skip connection.
The feature maps obtained from the first block are down- sampled using a max-pooling layer of kernel-size=2, and stride=2, to become the input of the next block. The above procedure is repeated thrice (i.e. for encoder part), till the feature map size is reduced to 16x32. After this the feature maps are upsampled using a max-unpooling layer with similar dimension as the max-pooling layer. The upsampled feature map is passed to the convolutional, relu, batch-normalization layer, and then to the max unpooling layer thrice, till the feature maps are of the same resolution as the initial feature map.Feature maps of same resolution from down-sampling and up-sampling layers are connected through residual dense skip connection in the up-sampling (decoder) path to incorporate both coarser and finer information. Finally the output is passed on to the softmax layer with 5 channels in order to implement 5 label classification (with 5 channels).

# Experiments

We have used Residual-Dense USegNet, and have compared it with baseline models like SegNet, USegNet and Unet. We
have also applied variants to USegnet, like Dense USegNet (In this, we have applied dense connections between each block i.e. 3 dense connections though in USegnet, we have only 1 dense skip connection) , Residual USegNet (In this instead of dense skip connection, we have applied residual skip connection.), Residual-Residual USegNet(In this we have applied residual connections between each block i.e. 3 residual skip connections though in USegnet, we have only 1 dense skip connection) ,USegNet with transConvolution(In this we have applied trans convolution for upsampling though in Usegnet we use unpooling for upsampling) , Residual-Residual USegNet with transConvolution(In this we have applied resid- ual connections between each block i.e. 3 residual skip connec- tions with trans convolution for upsampling). USegNet without skipConnection and with transConvolution(In this, we haven’t applied any skip connection but used transconvolution for upsampling). For all the models we have applied the same parameters and our model outperforms all.
Dice ratio has been the major metric for comparison among these models. Another important parameter was the confusion matrix for test images. Larger the number of correct fore- ground pixels, better are the results. Our dataset had original images, and also semantic segmentation of those images. We have done the preprocessing on those images, to form the ground truth. The implementation details remain the same as described above.
In our experiments, the test ac-curacies of almost all the models were similar, given the large number of background pixels correctly predicted. But our main focus is correctly predicting the foreground pixels, i.e. the dice ratio should be higher. For that we can refer the table and see that, our model USegNet Residual Dense outperforms all.

# Post-Processing :

From our architecture, we obtain the results which look like the one given in the figure 3. There are 5 channels correspond- ing to car, bicycle or rider, truck, person, motorcycle and are in this order only. If there is car in the image, the channel 1 will have convex hull results. If there is no car, then the 1st channel will be empty. See the figure 3. Now these channels are passed to convex hull algorithm, but before that we are doing gray scale conversion, image blurring, thresholding and are passing the converted images to find the contours using Chain Approx Simple method. If the area of contour is greater than certain value, we are bounding the real life image corresponding to the contour within the contour lines

#  Results :

We can see the results in Docs folder.

https://github.com/Kaamraan19064/ConvexHull-Object-Detection/tree/main/Docs

