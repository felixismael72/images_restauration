import cv2 as cv
import matplotlib.pyplot as plt


def restore_and_show_img_set(damaged_img, mask, in_painting_method):
    in_painted_img = cv.inpaint(damaged_img, mask, 15, in_painting_method)
    noise_reduced_img = cv.fastNlMeansDenoisingColored(in_painted_img, None, 10, 10, 7, 21)

    img = [damaged_img, mask, in_painted_img, noise_reduced_img]
    titles = ['damaged image', 'mask', 'in painted img', 'noise reduced img']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
        plt.imshow(img[i])
    plt.show()

    return noise_reduced_img


damaged_img1 = cv.imread("baby_with_drawing.jpg")
mask1 = cv.imread("baby_mask.png", cv.IMREAD_GRAYSCALE)
damaged_img2 = cv.imread("damaged.png")
mask2 = cv.imread("damaged_mask.png", cv.IMREAD_GRAYSCALE)
damaged_img3 = cv.imread("boy_in_damaged_photo.png")
mask3 = cv.imread("boy_in_damaged_photo_mask.png", cv.IMREAD_GRAYSCALE)
damaged_img4 = cv.imread("girl_in_damaged_photo.png")
mask4 = cv.imread("girl_in_damaged_photo_mask.png", cv.IMREAD_GRAYSCALE)

to_save_1 = restore_and_show_img_set(damaged_img1, mask1, cv.INPAINT_NS)
to_save_2 = restore_and_show_img_set(damaged_img2, mask2, cv.INPAINT_TELEA)
to_save_3 = restore_and_show_img_set(damaged_img3, mask3, cv.INPAINT_NS)
to_save_4 = restore_and_show_img_set(damaged_img4, mask4, cv.INPAINT_NS)

cv.imwrite("first_restored.png", to_save_1)
cv.imwrite("second_restored.png", to_save_2)
cv.imwrite("third_restored.png", to_save_3)
cv.imwrite("forth_restored.png", to_save_4)
