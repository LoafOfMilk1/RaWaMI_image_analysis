import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageChops

folder_path = r"C:\Users\Squishfolk\Documents\Gusty\Exp4LowDrop"
green_folder_path = r"C:\Users\Squishfolk\Documents\Gusty\TestImagesGreen"
diff_folder_path = r"C:\Users\Squishfolk\Documents\Gusty\TestImagesDiff"

# create green and difference folders if they don't exist
if not os.path.exists(green_folder_path):
    os.makedirs(green_folder_path)
if not os.path.exists(diff_folder_path):
    os.makedirs(diff_folder_path)

# loop through each HEIC file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tif"):
        # open the image and get the green channel
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            green_img = img.split()[1]

        # save the green channel as a new image in the green folder
        green_filename = f"{os.path.splitext(filename)[0]}_green.tif"
        green_file_path = os.path.join(green_folder_path, green_filename)
        green_img.save(green_file_path)

# function to calculate difference image and average pixel brightness
# function to calculate difference image and average pixel brightness
def calc_difference(start_index, step):
    # get the two images to calculate difference
    green_files = sorted(os.listdir(green_folder_path))
    start_file = green_files[start_index]
    end_index = start_index + step
    end_file = green_files[end_index]
    start_img_path = os.path.join(green_folder_path, start_file)
    end_img_path = os.path.join(green_folder_path, end_file)
    start_img = Image.open(start_img_path)
    end_img = Image.open(end_img_path)

    # calculate the difference image and save it
    diff_img = ImageChops.difference(start_img, end_img)
    diff_filename = f"diff_{start_file}_{end_file}"
    diff_file_path = os.path.join(diff_folder_path, diff_filename)
    diff_img.save(diff_file_path)

    # calculate the average pixel brightness of the difference image
    diff_pixels = diff_img.load()
    width, height = diff_img.size
    total_brightness = 0
    for x in range(width):
        for y in range(height):
            pixel_value = diff_pixels[x, y]
            total_brightness += pixel_value
    avg_brightness = total_brightness / (width * height)
    return avg_brightness

def pixel_height_brightness(start_index, step):
    # get the two images to calculate difference
    green_files = sorted(os.listdir(green_folder_path))
    start_file = green_files[start_index]
    end_index = start_index + step
    end_file = green_files[end_index]
    start_img_path = os.path.join(green_folder_path, start_file)
    end_img_path = os.path.join(green_folder_path, end_file)
    start_img = Image.open(start_img_path)
    end_img = Image.open(end_img_path)

    # calculate the difference image and save it
    diff_img = ImageChops.difference(start_img, end_img)
    diff_filename = f"diff_{start_file}_{end_file}"
    diff_file_path = os.path.join(diff_folder_path, diff_filename)
    diff_img.save(diff_file_path)

    # calculate the average pixel brightness of each row of the difference image
    diff_pixels = diff_img.load()
    width, height = diff_img.size
    row_brightnesses = []
    for y in range(height):
        total_brightness = 0
        for x in range(width):
            pixel_value = diff_pixels[x, y]
            total_brightness += pixel_value
        avg_brightness = total_brightness / width
        row_brightnesses.append(avg_brightness)
    return row_brightnesses, list(range(height))

def pixel_width_brightness(start_index, step):
    # get the two images to calculate difference
    green_files = sorted(os.listdir(green_folder_path))
    start_file = green_files[start_index]
    end_index = start_index + step
    end_file = green_files[end_index]
    start_img_path = os.path.join(green_folder_path, start_file)
    end_img_path = os.path.join(green_folder_path, end_file)
    start_img = Image.open(start_img_path)
    end_img = Image.open(end_img_path)

    # calculate the difference image and save it
    diff_img = ImageChops.difference(start_img, end_img)
    diff_filename = f"diff_{start_file}_{end_file}"
    diff_file_path = os.path.join(diff_folder_path, diff_filename)
    diff_img.save(diff_file_path)

    # calculate the average pixel brightness of each row of the difference image
    diff_pixels = diff_img.load()
    width, height = diff_img.size
    row_brightnesses = []
    for y in range(width):
        total_brightness = 0
        for x in range(height):
            pixel_value = diff_pixels[y, x]
            total_brightness += pixel_value
        avg_brightness = total_brightness / height
        row_brightnesses.append(avg_brightness)
    return row_brightnesses, list(range(width))


# calculate difference image and average brightness for specified start and step
# example values, replace with your desired inputs

def create_step_vector(start_value,starting_step,number_of_steps):
    steps_intensity_vector = []
    steps_vector = []
    for i in range(1,number_of_steps*starting_step):
        steps_intensity_vector.append(calc_difference(start_value,i))
        steps_vector.append(starting_step-1+i)
    return steps_intensity_vector,steps_vector,start_value

def create_start_vector(start_value,step,number_of_starts):
    start_intensity_vector = []
    start_vector = []
    for i in range(1,start_value*number_of_starts):
        start_intensity_vector.append(calc_difference(i,step))
        start_vector.append(start_value-1+i)
    return start_intensity_vector,start_vector,step


#step calculater to find aggregate brightness differences across larger ranges of images.
# Works better with less background noise. Used to generate plots 1 and 2
# x_steps = create_step_vector(1,1,8)[1]
# y_steps_1 = create_step_vector(1,1,8)[0]
# y_steps_2 = create_step_vector(2,1,8)[0]
# x_starts = create_start_vector(1,1,8)[1]
# y_starts_1 = create_start_vector(1,1,8)[0]

#perhaps unideal and inefficient naming convention for brightness vectors, used to generate plots 3 and 4
brightness_vector1, row_number_vector1 = pixel_height_brightness(0, 1)
brightness_vector2, row_number_vector2 = pixel_height_brightness(0, 2)
brightness_vector3, row_number_vector3 = pixel_height_brightness(0, 3)
brightness_vector4, row_number_vector4 = pixel_height_brightness(0, 4)
brightness_vector5, row_number_vector5 = pixel_width_brightness(0, 1)
brightness_vector6, row_number_vector6 = pixel_width_brightness(0, 2)
brightness_vector7, row_number_vector7 = pixel_width_brightness(0, 3)
brightness_vector8, row_number_vector8 = pixel_width_brightness(0, 4)

#boxcar smoothing function

def smoother(vector,box_pts):
    box = np.ones(box_pts)/box_pts
    vector_smooth = np.convolve(vector,box,mode='same')
    return  vector_smooth 

def offset(vector, offset):
     newvector = [x - offset for x in vector]
     return newvector

#boxcar parameter
points = 100

#uncomment thetwo blocks below to plot figs 1 and 2. 
# fig1, ax1 = plt.subplots()
# ax1.plot(x_steps, y_steps_1, label='From Drop 1')
# ax1.plot(x_steps, y_steps_2, label='From Drop 2')
# ax1.set_xlabel('Step Increment')
# ax1.set_ylabel('Average Pixel Brightness')
# ax1.legend()
# ax1.set_title('Average Aggregate Pixel Brightness as a Function of Step Increment')

# fig2, ax2 = plt.subplots()
# ax2.plot(x_starts, y_starts_1, label='Step Increment 1')
# ax2.set_xlabel('Starting Drop')
# ax2.set_ylabel('Average Pixel Brightness')
# ax2.legend()
# ax2.set_title('Average Aggregate Pixel Brightness as a Function of Starting Drop')



# uncomment for to delta across drops (example where vec1 = drop 1-22, vec2 = drop 1-90, vec4 = drop 1-360)
fig3, ax3 = plt.subplots()
ax3.plot(row_number_vector1, offset(smoother(brightness_vector1,points), 0.4), label='From drop 1-22 (-.4 y-offset)')
ax3.plot(row_number_vector1, np.subtract(smoother(brightness_vector2,points),smoother(brightness_vector1,points)), label='From drop 22-90')
ax3.plot(row_number_vector1, np.subtract(smoother(brightness_vector4,points),smoother(brightness_vector2,points)), label='From drop 90-360')

plt.xlim(420, 1450)
ax3.set_xlabel('Row Number (Starting at Top of Image)')
ax3.set_ylabel('Average Brightness Difference')
ax3.legend()
ax3.set_title('Average Row Pixel Brightness as a Function of Height')

# uncomment for standard avg brightness as function of height calc (standard incrementation)
# fig3, ax3 = plt.subplots()
# ax3.plot(row_number_vector1, smoother(brightness_vector1,points), label='From drop 1-22')
# ax3.plot(row_number_vector2, smoother(brightness_vector2,points), label='From drop 1-45')
# ax3.plot(row_number_vector3, smoother(brightness_vector3,points), label='From drop 1-68')
# ax3.plot(row_number_vector4, smoother(brightness_vector4,points), label='From drop 1-90')
# plt.xlim(550, 1400)
# plt.ylim(0.9,2)
# ax3.set_xlabel('Row Number (Starting at Top of Image)')
# ax3.set_ylabel('Avg. Brightness')
# ax3.legend()
# ax3.set_title('Avg. Row Pixel Brightness as a Function of Height')

#uncomment for to delta across drops (example where vec5 = drop 1-22, vec6 = drop 1-90, vec8 = drop 1-360)
fig4, ax4 = plt.subplots()
ax4.plot(row_number_vector5, offset(smoother(brightness_vector5,points), 0.5), label='From drop 1-22 (-.5 y-offset)')
ax4.plot(row_number_vector5, np.subtract(smoother(brightness_vector6,points),smoother(brightness_vector5,points)), label='From drop 22-90')
ax4.plot(row_number_vector5, np.subtract(smoother(brightness_vector8,points),smoother(brightness_vector6,points)), label='From drop 90-360')
plt.xlim(900, 2600)
ax4.set_xlabel('Column Number (Starting at Left of Image)')
ax4.set_ylabel('Average Brightness Difference')
ax4.legend()
ax4.set_title('Average Column Pixel Brightness as a Function of Width')


#uncomment for standard avg brightness as function of width calc (standard incrementation)
# fig4, ax4 = plt.subplots()
# ax4.plot(row_number_vector5, smoother(brightness_vector5,points), label='From drop 1-22')
# ax4.plot(row_number_vector6, smoother(brightness_vector6,points), label='From drop 1-45')
# ax4.plot(row_number_vector7, smoother(brightness_vector7,points), label='From drop 1-68')
# ax4.plot(row_number_vector8, smoother(brightness_vector8,points), label='From drop 1-90')
# ax4.set_xlabel('Column Number (Starting at Left of Image)')
# ax4.set_ylabel('Average Brightness')
# ax4.legend()
# ax4.set_title('Average Column Pixel Brightness as a Function of Width')
# plt.xlim(400, 3200)
# plt.ylim(.5,2.2)

plt.show()