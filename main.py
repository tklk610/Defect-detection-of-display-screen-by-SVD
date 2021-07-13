# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from numpy import linalg


src_path = r"C:\Users\lekang\PycharmProjects\ImageProcess\SVD\image"
tar_path = r"C:\Users\lekang\PycharmProjects\ImageProcess\SVD\data"


print(src_path)

def k_value(r, *args) :
    new  = list(args)
    mean = np.mean(new)
    std  = np.std(new, ddof = 1)

    for i in range(len(new)) :
        new[i] = (new[i] - mean) / std


    sigma_date = []
    num_data   = []

    for i in range(len(new)) :
        if i == 0 :
            pass
        else :
            sigma = new[i-1] - new[i]
            #print(sigma)
            sigma_date.append(sigma)

            j = i-1
            num_data.append(j)

    plt.plot(num_data, sigma_date)        # 绘制输入的结果和值的结果
    plt.title("Sigma data", fontsize=15)  # 标题
    plt.xlabel("num", fontsize=15)        # 横坐标
    plt.ylabel("sigma", fontsize=15)      # 纵坐标
    plt.show()  # 显示
    #cv2.waitKey()


    for i in range(len(sigma_date)) :
        if sigma_date[i] <= r :
            print("Singular value is %g,number is %g" %(sigma_date[i], i))
            return i




def svd(src_path, tar_path, r) :
    #print("path = %s" %(path))
    for file in os.listdir(src_path) :
        file_path = os.path.join(src_path, file)


        if os.path.isdir(file_path):
            os.list_dir(file_path)
        else:
            img = cv2.imread(file_path)
            print(img.shape)
            ch = img.shape[2]
            print(ch)

            if ch == 3:       # color image with 3 channels
                print("This is color image!")
                image = img

                for ch in range(3) :
                    img_index    = img[:, :, ch]
                    U, sigma, VT = linalg.svd(img_index)
                    u_shape      = U.shape[0]
                    vt_shape     = VT.shape[0]

                    # k = k_value(r, *sigma)

                    k = 5  # # set k value artificially

                    if k <0 :
                        raise Exception("k should lager than 0!", k)

                    # print(U.shape)
                    # print(sigma.shape)
                    # print(VT.shape)

                    sigma[0:k] = 0
                    #print(sigma)

                    new_sigma = np.zeros((u_shape, vt_shape))
                    index = u_shape if u_shape < vt_shape else vt_shape

                    for row in range(index) :
                        for col in range(index) :
                            if row == col :
                                new_sigma[row][col] = sigma[row]

                    print(new_sigma.shape)

                    # 重构矩阵
                    # dig = np.mat(np.eye(num) * sigma[:])  # 获得对角矩阵
                    # dim = data.T * U[:,:count] * dig.I      # 降维
                    image[:, :, ch] = np.dot(np.dot(U[:, :], new_sigma), VT[:, :])  # 重构


                print(image.shape)
                new_file = file[:-4] + '_' + str(ch) + '_' + str(k)  + '.bmp'
                #new_file = file[:-5] + '_' + str(ch) + '_' + str(k) + '.tiff'
                new_name = os.path.join(tar_path, new_file)
                cv2.imwrite(new_name, image)

            else :  # gray image
                print("This is gray image!")
                U, sigma, VT = linalg.svd(img)
                u_shape      = U.shape[0]
                vt_shape     = VT.shape[0]

                k = k_value(r, sigma)

                # k = 17   # set k value artificially

                if k < 0:
                    raise Exception("k should lager than 0!", k)

                print(U.shape)
                print(sigma.shape)
                print(VT.shape)
                sigma[0:k] = 0

                new_sigma = np.zeros((u_shape, vt_shape))
                index = u_shape if u_shape < vt_shape else vt_shape

                for row in range(index):
                    for col in range(index):
                        if row == col:
                            new_sigma[row][col] = sigma[row]

                print(new_sigma.shape)

                image = np.dot(np.dot(U[:, :], new_sigma), VT[:, :])  # 重构
                print(image.shape)
                #new_file = file[:-4] + '_' + str(k) + '.bmp'
                new_file = file[:-5] + '_' + str(k) + '.tiff'
                new_name = os.path.join(tar_path, new_file)
                cv2.imwrite(new_name, image)


svd(src_path, tar_path, 0.05)


print('Finished')
