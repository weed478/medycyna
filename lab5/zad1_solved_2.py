import SimpleITK as sitk
import matplotlib.pyplot as plt

def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
    global iteration_number
    central_indexes = [int(i / 2) for i in fixed.GetSize()]

    moving_transformed = sitk.Resample(moving, fixed, transform,
                                       sitk.sitkLinear, 0.0,
                                       moving_image.GetPixelIDValue())
    # extract the central slice in xy, xz, yz and alpha blend them
    combined = [fixed[:, :, central_indexes[2]] + -
    moving_transformed[:, :, central_indexes[2]],
                fixed[:, central_indexes[1], :] + -
                moving_transformed[:, central_indexes[1], :],
                fixed[central_indexes[0], :, :] + -
                moving_transformed[central_indexes[0], :, :]]

    # resample the alpha blended images to be isotropic and rescale intensity
    # values so that they are in [0,255], this satisfies the requirements
    # of the jpg format
    print(iteration_number, ": ", registration_method.GetMetricValue());
    combined_isotropic = []
    for img in combined:
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        min_spacing = min(original_spacing)
        new_spacing = [min_spacing, min_spacing]
        new_size = [int(round(original_size[0] * (original_spacing[0] / min_spacing))),
                    int(round(original_size[1] * (original_spacing[1] / min_spacing)))]
        resampled_img = sitk.Resample(img, new_size, sitk.Transform(),
                                      sitk.sitkLinear, img.GetOrigin(),
                                      new_spacing, img.GetDirection(), 0.0,
                                      img.GetPixelIDValue())
        combined_isotropic.append(sitk.Cast(sitk.RescaleIntensity(resampled_img),
                                            sitk.sitkUInt8))
    # tile the three images into one large image and save using the given file
    # name prefix and the iteration number
    sitk.WriteImage(sitk.Tile(combined_isotropic, (1, 3)),
                    file_name_prefix + format(iteration_number, '03d') + '.jpg')
    iteration_number += 1


# read the images
fixed_image = sitk.ReadImage("data/nativeFixed.mhd", sitk.sitkFloat32)
moving_image = sitk.ReadImage("data/tetniceMoving1.mhd", sitk.sitkFloat32)

transform = sitk.CenteredTransformInitializer(fixed_image,
                                                      moving_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)

# multi-resolution rigid registration using Mutual Information
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation()
registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
registration_method.SetMetricSamplingPercentage(1)
registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsGradientDescent(learningRate=0.5,
                                                  numberOfIterations=100,
                                                  convergenceMinimumValue=1e-6,
                                                  convergenceWindowSize=5)


registration_method.SetInitialTransform(transform)



# # Utwórz transformację FFD
# transform = sitk.BSplineTransformInitializer(fixed_image, [3, 3, 3])
#
# # Multi-resolution registration
# registration_method = sitk.ImageRegistrationMethod()
# registration_method.SetMetricAsMattesMutualInformation()
# registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
# registration_method.SetMetricSamplingPercentage(1)
# registration_method.SetInterpolator(sitk.sitkLinear)
# registration_method.SetOptimizerAsGradientDescent(learningRate=0.05,
#                                                       numberOfIterations=100,
#                                                       convergenceMinimumValue=1e-6,
#                                                       convergenceWindowSize=20)
# registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
# registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
# registration_method.SetInitialTransform(transform)


import time
# Pomiar czasu przed rejestracją
start_time = time.time()

# add iteration callback, save central slice in xy, xz, yz planes
global iteration_number
iteration_number = 0
registration_method.AddCommand(sitk.sitkIterationEvent,
                               lambda: save_combined_central_slice(fixed_image,
                                                                   moving_image,
                                                                   transform,
                                                                   'output/iteration'))
print("Initial metric: ", registration_method.MetricEvaluate(fixed_image, moving_image))
final_transform = registration_method.Execute(fixed_image, moving_image)

# Pomiar czasu po rejestracji
end_time = time.time()


print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
print("Metric value after  registration: ", registration_method.GetMetricValue())

# Różnica czasów
registration_time = end_time - start_time
print("Time taken for registration: {:.2f} seconds".format(registration_time))

sitk.WriteTransform(final_transform, 'output/ct2mrT1.tfm')


import numpy as np

# Przekształcenie obrazu poruszającego
moving_image_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue())

# Obliczenie różnicy między obrazami
difference_image = sitk.Abs(fixed_image - moving_image_transformed)

# Konwersja obrazu różnicowego do tablicy numpy
difference_array = sitk.GetArrayViewFromImage(difference_image)

# # Wyświetlenie obrazu przy użyciu matplotlib
# plt.imshow(difference_array[:, :, 0], cmap='gray')  # Wybieramy tylko pierwszy kanał, jeśli jest trzeci
# plt.axis('off')  # Wyłączenie osi
# plt.show()


# Przekształcenie obrazu poruszającego
moving_image_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue())

# Obliczenie różnicy między obrazami
difference_image = sitk.Abs(fixed_image - moving_image_transformed)


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def generate_checkerboard_images(fixed, moving):
    # Pobranie rozmiarów obrazów
    size_x = fixed.GetWidth()
    size_y = fixed.GetHeight()

    # Utworzenie pustego obrazu szachownicy
    checkerboard = sitk.Image(size_x, size_y, sitk.sitkUInt8)

    # Iteracja po blokach i łączenie ich naprzemiennie
    for y in range(0, size_y, block_size):
        for x in range(0, size_x, block_size):
            # Sprawdzenie, czy aktualny blok powinien być z obrazu fixed czy moving
            if (x // block_size + y // block_size) % 2 == 0:
                checkerboard[x:x+block_size, y:y+block_size] = fixed[x:x+block_size, y:y+block_size]
            else:
                checkerboard[x:x+block_size, y:y+block_size] = moving[x:x+block_size, y:y+block_size]

    return checkerboard


# Rozmiar bloku - można dostosować w zależności od potrzeb
block_size = 32

# Przekonwertowanie obrazów na typ piksela 8-bit unsigned integer
fixed = sitk.Cast(fixed_image[:, :, 0], sitk.sitkUInt8)
moving = sitk.Cast(moving_image_transformed[:, :, 0], sitk.sitkUInt8)

# Wygenerowanie obrazu szachownicy
checkerboard_image = generate_checkerboard_images(fixed, moving)

# Wyświetlenie obrazu szachownicy
plt.imshow(sitk.GetArrayFromImage(checkerboard_image), cmap='gray')
plt.axis('off')
plt.show()


# # Wizualizacja 3D
# def plot_3d(image, title):
#     # Przekształcenie obrazu do tablicy numpy
#     image_array = sitk.GetArrayFromImage(image)
#
#     # Uzyskanie indeksów niezerowych pikseli
#     non_zero_indices = np.where(image_array > 0)
#
#     # Określenie granic obrazu
#     x_min, x_max = non_zero_indices[0].min(), non_zero_indices[0].max()
#     y_min, y_max = non_zero_indices[1].min(), non_zero_indices[1].max()
#     z_min, z_max = non_zero_indices[2].min(), non_zero_indices[2].max()
#
#     # Utworzenie figur
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Wyświetlenie obrazu 3D
#     ax.scatter(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2], c=image_array[non_zero_indices], cmap='gray', marker='.')
#
#     # Ustawienie tytułu i etykiet osi
#     ax.set_title(title)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # Ustawienie granic wykresu na podstawie obrazu
#     ax.set_xlim([x_min, x_max])
#     ax.set_ylim([y_min, y_max])
#     ax.set_zlim([z_min, z_max])
#
#     plt.show()
#
# # Wyświetlenie obrazu przed rejestracją
# plot_3d(fixed_image, title='Fixed Image')
#
# # Wyświetlenie obrazu po rejestracji
# plot_3d(moving_image_transformed, title='Transformed Moving Image')
#
# from scipy.spatial import cKDTree
#
# # Funkcja do weryfikacji wyników za pomocą histogramu odległości
# def calculate_point_distances(fixed, moving_transformed, num_points=100):
#     fixed_points = sitk.GetArrayFromImage(fixed)
#     moving_points = sitk.GetArrayFromImage(moving_transformed)
#
#     # Losowy wybór punktów
#     fixed_indices = np.random.choice(fixed_points.size, size=num_points, replace=False)
#     moving_indices = np.random.choice(moving_points.size, size=num_points, replace=False)
#
#     fixed_coords = np.array(np.unravel_index(fixed_indices, fixed_points.shape)).T
#     moving_coords = np.array(np.unravel_index(moving_indices, moving_points.shape)).T
#
#     # Obliczanie odległości
#     tree = cKDTree(moving_coords)
#     distances, _ = tree.query(fixed_coords)
#
#     return distances
#
# # Obliczanie i wyświetlanie histogramu odległości
# distances = calculate_point_distances(fixed_image, moving_image_transformed)
# plt.hist(distances, bins=20, alpha=0.75)
# plt.xlabel('Odległość')
# plt.ylabel('Liczba punktów')
# plt.title('Histogram odległości pomiędzy odpowiadającymi sobie punktami')
# plt.show()