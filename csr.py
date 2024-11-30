import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from google.colab import files
from scipy.optimize import curve_fit
from math import atan, degrees
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def contour_length(contour):
    return cv.arcLength(contour, True)

def preprocessing(img):
    denoised_img = cv.GaussianBlur(img, (5, 5), 0)

    # define region of interest (roi) containing only required edges
    x_start, y_start = 100, 0
    x_end, y_end = 500, 275
    roi = denoised_img[y_start:y_end, x_start:x_end]

    # find and display edges
    edges = cv.Canny(roi, 40,80, L2gradient = False)
    plt.subplot(121), plt.imshow(denoised_img, cmap = 'seismic')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap = 'seismic')
    plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
    plt.show()

    # find contours and sort based on length (perimeter)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=contour_length, reverse=True)

    return contours


# def linear_func(x, a, b):
#     return a * x + b

def compute_slope(img, cnt, start, end):
    # determine edge and approximation
    cnt = cnt[start:end]
    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt,epsilon,True)

    # bgr_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # cv.drawContours(bgr_image, cnt,-1,(0,255,0),3) # green
    # cv.drawContours(bgr_image, [cnt], -1,(255,0,0), 3) # blue
    # cv.drawContours(bgr_image, [cnt], -1, (0,0,255), 3) # red
    # cv2_imshow(bgr_image)
    # print('\n')

    x_coords = [point[0][0] for point in approx]
    y_coords = [point[0][1] for point in approx]
    x_coords = [x+100 for x in x_coords]

    linear_func = lambda x, a, b: a * x + b
    popt, _ = curve_fit(linear_func, x_coords, y_coords)
    plt.plot(x_coords, linear_func(np.array(x_coords), *popt), color='red')
    plt.title('Detected Shock and Fitted Line')

    h, _ = img.shape
    y_coords_real = [(h-y) for y in y_coords]
    popt, _ = curve_fit(linear_func, x_coords, y_coords_real)
    slope = popt[0]

    return(atan(slope))


def conical_shock_calculator(cone_angle, wave_angle):
    try:
        # load web driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)

        # binary search between [1.0, 7.0], with elements at steps of stepsize (1e-10)
        step_size = 1e-10
        start = 1.0
        end = 7.0
        num_eles = int((end - start)/step_size)

        l = 0
        r = num_eles-1
        mid = int((l+r)/2)+1
        m1 = start + step_size * mid

        while(True):
            # get webpage from url and select the conical shock relations form out of the 6 forms on the page
            driver.get('https://devenport.aoe.vt.edu/aoe3114/calcbody.html')
            forms = driver.find_elements(By.TAG_NAME, 'form')
            csr_form = forms[3]

            # send the cone_angle to the cone angle input field
            input_cone_angle = csr_form.find_element(By.NAME, 'a')
            input_cone_angle.clear()
            input_cone_angle.send_keys(degrees(cone_angle))

            # send current value of M1 to the M1 input field
            input_m1 = csr_form.find_element(By.NAME, 'm')
            input_m1.clear()
            input_m1.send_keys(m1)

            calc_buttons = csr_form.find_elements(By.XPATH, "//input[@type='button']")
            calc_buttons[3].click()

            beta_result = csr_form.find_element(By.NAME, 'beta')
            beta_val = float(beta_result.get_attribute('value'))

            wave_angle_deg = degrees(wave_angle)
            if (abs(beta_val - wave_angle_deg) < 1e-7):
                break
            else:
                if beta_val < wave_angle_deg:
                    r = mid
                else:
                    l = mid

                mid = int((l+r)/2)+1
                m1 = start + step_size * mid

        return m1

    finally:
        driver.quit()


if __name__ == '__main__':
    # load image file
    print('Upload the image file:')
    uploaded_files = files.upload()
    assert len(uploaded_files) == 1, "Upload only one file"
    file_name = list(uploaded_files.keys())[0]

    print(f'Image {file_name}\n')
    img = cv.imread(f'{file_name}', cv.IMREAD_GRAYSCALE)
    assert img is not None

    contours = preprocessing(img)
    cone_angle = compute_slope(img, contours[0], 0, min(len(contours[0]), 100))
    wave_angle = compute_slope(img, contours[1], 0, min(len(contours[1]), 100))
    mach_number = conical_shock_calculator(cone_angle, wave_angle)

    print('\n')
    plt.imshow(img, cmap='gray')
    plt.show()

    print('\nCone angle:', degrees(cone_angle), '\nWave angle:', degrees(wave_angle), '\nMach number:', mach_number, '\n')

