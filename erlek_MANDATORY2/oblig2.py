import numpy as np
import matplotlib.pyplot as plt 

#Function for plotting each GLCM image that we have to pick which to choose from
def Analyze_textures(number):
    texturedx0dymin1 = np.genfromtxt(f"oblig2-python/texture{number}dx0dymin1.txt", dtype=float, delimiter=",")
    texturedx1dy0 = np.genfromtxt(f"oblig2-python/texture{number}dx1dy0.txt", dtype=float, delimiter=",")
    texturedx1dymin1 = np.genfromtxt(f"oblig2-python/texture{number}dx1dymin1.txt", dtype=float, delimiter=",")
    texturedxmin1dymin1 = np.genfromtxt(f"oblig2-python/texture{number}dxmin1dymin1.txt", dtype=float, delimiter=",")

    #Plotting for visualization
    fig = plt.figure()
    columns = 2
    rows = 2
    GLCM = [texturedx0dymin1,texturedx1dy0,texturedx1dymin1,texturedxmin1dymin1]
    for i in range(1,5):
        fig.add_subplot(rows, columns, i)
        plt.imshow(GLCM[i-1])
    plt.tight_layout()
    plt.savefig(f"figures/GLCM{number}")
    plt.show()

#function for creating our features from a GLCM 
def Create_features(GLCM):
    features = np.zeros(4)
    sum = 0
    for i in range(16):
        if i < 8:
            for j in range(8):
                features[0] += GLCM[i,j]
                features[1] += GLCM[i,j+8]
        if i > 8:
            for j in range(8): 
                features[2] += GLCM[i,j]
                features[3] += GLCM[i,j+8]  
    return features

#Function for making GLCM 
def MakeGLCM(im, dist=1, degree=0, L=16, preset_graylevels = None):
    #requantize
    if L != None:
        im = np.floor(im * (L/255)) * (255/L)

    i_jump = 0
    j_jump = 0

    if degree == 0:
        j_jump = -1
    if degree == 45:
        i_jump = 1
        j_jump = -1
    if degree == 90:
        i_jump = 1
    if degree == 135:
        i_jump = 1
        j_jump = 1
    if degree == 180:
        j_jump = 1
    if degree == 225:
        i_jump = 1
        i_jump = -1
    if degree == 270:
        i_jump = -1
    if degree == 315:
        i_jump = -1
        j_jump = -1
    
    i_start = 1 if i_jump == -1 else 0
    i_end = -1 if i_jump == 1 else 0
    j_start = 1 if j_jump == -1 else 0
    j_end = -1 if j_jump == 1 else 0

    if (preset_graylevels).all() == None:
        gray_levels = np.unique(im)
    else:
        gray_levels = preset_graylevels
    map1 = {}
    map2 = {}
    for i in range(len(gray_levels)):
        map1[i] = gray_levels[i]
        map2[gray_levels[i]] = i

    num_gray_levels = len(gray_levels)
    
    Q = np.zeros((num_gray_levels,num_gray_levels))

    for i in range(i_start*dist, len(im) + i_end*dist):
        for j in range(j_start*dist, len(im[0]) + j_end*dist):
            val1 = im[i,j]
            val2 = im[i+i_jump*dist,j+j_jump*dist]
            Q[map2[val1],map2[val2]] += 1
    P = Q/(sum(sum(Q)))
    return P

#Function for making subplots consisting of multiple images
def make_Subplots(list, name=None):

    #Plotting for visualization
    fig = plt.figure()
    columns = 2
    rows = 2
    for i in range(1,5):
        fig.add_subplot(rows, columns, i)
        plt.imshow(list[i-1])
    plt.tight_layout()
    if name != None:
        plt.savefig(name)
    plt.show()


#Function for creating a feature matrix
def feature_matrix(image, size=31, L=None, d = 1, theta = 0, gray_levels = None):
    if L != None:
        image = np.floor(image * (L/255)) * (255/L)

    #out = np.ones_like(image, float)
    #gray_levels = np.unique(image)
    out = np.zeros((image.shape[0], image.shape[1], 4))
    pad = int(size/2)
    image = np.pad(image, pad, 'reflect')
    
    for i in range(pad, len(image)-pad):
        for j in range(pad, len(image[0])-pad):
            P = MakeGLCM(image[i-pad:i+pad,j-pad:j+pad], d, theta, L=None, preset_graylevels=gray_levels)
            features = Create_features(P)
            out[i-pad,j-pad] = features
    return out


class GausianClassifier:
    
    def fit(self, mask, features):
        self.features = features
        self.mask = mask
        num_features = len(features)
        num_classes = len(np.unique(mask)) - 1 #-1 since 0 is not a class
        num_pixels = np.zeros(num_classes)

        mu = np.zeros((num_classes, num_features))
        mu = np.matrix(mu)
  
        self.num_classes = num_classes
        self.num_features = num_features

        #Calculating mu
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                correct_class = int(mask[i,j])
                if correct_class != 0:
                    num_pixels[correct_class - 1] += 1
                    mu[correct_class - 1] += features[:,i,j]
        
        #normalizing mu
        for i in range(num_classes):
            mu[i] /= num_pixels[i]
        
        self.mu = mu 

        #Making sigma
        sigma = np.zeros((num_classes, num_features, num_features))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                correct_class = int(mask[i,j])
                if correct_class != 0:
                    sigma[correct_class - 1] += (features[:,i,j] - mu[correct_class - 1]).T*(features[:,i,j] - mu[correct_class - 1])
        
        #normalizing sigma
        for i in range(num_classes):
            sigma[i] /= num_pixels[i]
        
        self.sigma = sigma

        self.num_pixels = num_pixels


    def predict(self):
        mask = self.mask
        pred_matrix = np.zeros_like(self.mask)
        prob_arr = np.zeros(self.num_classes)

        #Probability for each class
        p_w = np.ones(self.num_classes)
        p_w *= self.num_pixels/np.sum(self.num_pixels)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                x = self.features[:,i,j]
                for k in range(self.num_classes):
                    first_part = 1/(np.sqrt(2*np.pi) * np.sqrt(np.linalg.det(self.sigma[k])))
                    #first_part = 1/(np.sqrt(2*np.pi) * np.sqrt(np.sum(np.sum(self.sigma[k])))) 
                    second_part = np.exp(-0.5 * (x - self.mu[k]) @ np.linalg.pinv(self.sigma[k]) @ (x - self.mu[k]).T) 
                    prob_arr[k] = first_part * second_part * p_w[k]
                pred_matrix[i,j] = np.argmax(prob_arr) + 1
        return pred_matrix

if __name__ == "__main__":
    #To analyze the different GLCMs
    #for i in range(1,5):
    #    Analyze_textures(i)

    ##
    # when analyzing we see that in texture 4, every GLCM picture is the same no matter
    # which direction we choose. Looking at all the textures in combination, we see that the
    # top right GLCM image is very different from each texture, so we choose this direction to work with, 
    # which is Angle 0, dx1dy0. The other options will result in texture 1 and 2 being too similar.  


    #Testing features
    def Test_features():
        for i in range(1,5):
            Chosen_GLCM = np.genfromtxt(f"oblig2-python/texture{i}dx1dy0.txt", dtype=float, delimiter=",")
            #Normalizing
            Chosen_GLCM = Chosen_GLCM/np.sum(np.sum(Chosen_GLCM))

            features = Create_features(Chosen_GLCM)
            print(features) 

    #Test_features()   
    ##
    # Outprint was:
    # [0.44053309 0.09730392 0.06081495 0.31779259]
    # [0.42009804 0.07071844 0.03501072 0.37784161]
    # [0.64474571 0.11894914 0.08148744 0.0932598 ]
    # [0.86672794 0.03900123 0.0178845  0.03975184]
    # As we can see, the features are able to seperate well between the textures,
    # but 1 and 2 are a bit alike. However, i think that as we see in feature 2 and 3 on these
    # textures, it seems the difference is large enough to be able to seperate between them. 
    # Therefore i don't think any further subdividing of the features is needed (but i might be wrong).
    # Also, since some of the quadrants are very similar, i think we need all four to be able
    # to seperate between the textures.
    # However, if we were to pick away one of them, it would be number 2 since 
    # the values for each texture are very similar.


    #Looking at training data
    train = np.genfromtxt(f"oblig2-python/mosaic1_train.txt", dtype=float, delimiter=",")
    train = np.floor(train * (16/255)) * (255/16)
    gray_levels = np.unique(train)

    texture1 = train[0:256, 0:256]
    texture2 = train[0:256, 256:512]
    texture3 = train[256:512, 0:256]
    texture4 = train[256:512, 256:512]
    textures = [texture1, texture2, texture3, texture4]

    #Function for teting the features by making feature images to observe
    def test_features():
        feature_im1 = feature_matrix(texture1, gray_levels=gray_levels)
        print("done with texture 1")
        feature_im2 = feature_matrix(texture2, gray_levels=gray_levels)
        print("done with texture 2")
        feature_im3 = feature_matrix(texture3, gray_levels=gray_levels)
        print("done with texture 3")
        feature_im4 = feature_matrix(texture4, gray_levels=gray_levels)
        print("done with texture 4")


        for i in range(4):
            new_list = [feature_im1[:,:,i], feature_im2[:,:,i], feature_im3[:,:,i], feature_im4[:,:,i]]
            make_Subplots(new_list, name=f"figures/feature{i}_img")
    
    #test_features()
    ##
    # Looking at the feature images, there is no clear indicaion that we can use 
    # a few of them and not all, therefore i choose to use all features for our classification.

    #Function for creating a 2d data  set which consists of sets of features 
    #and given texture class for those features.
    #REMOVE LATER
    def create_data():
        feature_im1 = feature_matrix(texture1, gray_levels=gray_levels)
        print("done with texture 1")
        feature_im2 = feature_matrix(texture2, gray_levels=gray_levels)
        print("done with texture 2")
        feature_im3 = feature_matrix(texture3, gray_levels=gray_levels)
        print("done with texture 3")
        feature_im4 = feature_matrix(texture4, gray_levels=gray_levels)
        print("done with texture 4")

        feature_im1 = feature_im1.reshape((feature_im1.shape[0] * feature_im1.shape[1]), feature_im1.shape[2])
        feature_im2 = feature_im2.reshape((feature_im2.shape[0] * feature_im2.shape[1]), feature_im2.shape[2])
        feature_im3 = feature_im3.reshape((feature_im3.shape[0] * feature_im3.shape[1]), feature_im3.shape[2])
        feature_im4 = feature_im4.reshape((feature_im4.shape[0] * feature_im4.shape[1]), feature_im4.shape[2])

        shape1 = feature_im1.shape[0]
        shape2 = feature_im2.shape[0]
        shape3 = feature_im3.shape[0]
        shape4 = feature_im4.shape[0]

        new_data = np.zeros((feature_im1.shape[0]+feature_im2.shape[0]+feature_im3.shape[0]+feature_im4.shape[0], feature_im1.shape[1] + 1)) 
        new_data[0:shape1,0:4] = feature_im1 
        new_data[0:shape1,4] = 1  
        new_data[shape1:shape1+shape2,0:4] = feature_im2 
        new_data[shape1:shape1+shape2,4] = 2
        new_data[shape1+shape2:shape1+shape2+shape3, 0:4] = feature_im3
        new_data[shape1+shape2:shape1+shape2+shape3, 4] = 3
        new_data[shape1+shape2+shape3:shape1+shape2+shape3+shape4 ,0:4] = feature_im4
        new_data[shape1+shape2+shape3:shape1+shape2+shape3+shape4 ,4] = 4

        np.savetxt('train_data', new_data, delimiter=',')

    #create_data()
    
    #Function for making and saving feature matrices we need for later
    #This is done simply to save time on later testing
    def save_FeatureMatrix():
        image = np.genfromtxt(f"oblig2-python/mosaic3_test.txt", dtype=float, delimiter=",")
        image = np.floor(image * (16/255)) * (255/16)
        gray_levels = np.unique(image)
        f_matrix = feature_matrix(image, gray_levels=gray_levels)
        np.savetxt('3Feature1Matrix.txt', f_matrix[:,:,0], delimiter=',')
        np.savetxt('3Feature2Matrix.txt', f_matrix[:,:,1], delimiter=',')
        np.savetxt('3Feature3Matrix.txt', f_matrix[:,:,2], delimiter=',')
        np.savetxt('3Feature4Matrix.txt', f_matrix[:,:,3], delimiter=',')
    
    #save_FeatureMatrix()

    #Function for making prediction matrix and visualizing it.
    #Will also store the prediction matrix in a .txt file for later use.
    def visualize_prediction():
        mask = np.genfromtxt(f"oblig2-python/mask3_mosaic3_test.txt", dtype=float, delimiter=",")
        FMatrix1 = np.genfromtxt(f"3Feature1Matrix.txt", dtype=float, delimiter=",")
        FMatrix2 = np.genfromtxt(f"3Feature2Matrix.txt", dtype=float, delimiter=",")
        FMatrix3 = np.genfromtxt(f"3Feature3Matrix.txt", dtype=float, delimiter=",")
        FMatrix4 = np.genfromtxt(f"3Feature4Matrix.txt", dtype=float, delimiter=",")
        FList = [FMatrix1, FMatrix2, FMatrix3, FMatrix4]
        FList = np.array(FList)
        model = GausianClassifier()
        model.fit(mask, FList)

        pred_matrix = model.predict()
        np.savetxt("PredMatrix3.txt", pred_matrix, delimiter=',')
        plt.imshow(pred_matrix)
        plt.savefig("figures/prediction_mask3.png")
        plt.show()

    #visualize_prediction()
    
    #Function for making confusion matrix
    def MakeConfusionMatrix():
        overall_accuracy = 0
        pred_matrix = np.genfromtxt(f"PredMatrix1.txt", dtype=int, delimiter=",")
        num_classes = len(np.unique(pred_matrix))
        confusionMatrix = np.zeros((num_classes, num_classes))
        for i in range(pred_matrix.shape[0]):
            for j in range(pred_matrix.shape[1]):
                if i < 256:
                    if j < 256:
                        correct_class = 1
                    else:
                        correct_class = 2
                else:
                    if j < 256:
                        correct_class = 3
                    else: 
                        correct_class = 4
                pred_class = pred_matrix[i,j]
                confusionMatrix[correct_class-1, pred_class-1] += 1
                if pred_class == correct_class:
                    overall_accuracy += 1
                

            
        confusionMatrix /= (pred_matrix.shape[0] * pred_matrix.shape[1])/400

        print(confusionMatrix)
        print(f"overall accuracy = {overall_accuracy/(pred_matrix.size)}")

    MakeConfusionMatrix()    



    
