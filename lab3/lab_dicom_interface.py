import vtk

import tkinter as tk
from tkinter import filedialog

# Funkcja do wczytywania danych DICOM
def loadDICOMData():
    root = tk.Tk()
    root.withdraw()  # Ukrycie okna głównego
    directory = filedialog.askdirectory(title="Wybierz katalog z danymi DICOM")  # Okno dialogowe do wyboru katalogu
    print('Directory: ', directory)
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()
    return reader.GetOutput()


# Funkcja do aktualizacji danych po wczytaniu nowej serii
def updateData(newImageData):
    global imageData, shiftScaleFilter, contourFilter
    imageData = newImageData
    shiftScaleFilter.SetInputData(imageData)
    shiftScaleFilter.Update()
    contourFilter.SetInputConnection(shiftScaleFilter.GetOutputPort())
    contourFilter.Update()
    sliderRep.SetMaximumValue(imageData.GetDimensions()[2])
    sliderRep.SetValue(imageData.GetDimensions()[2] / 2)
    renWin.Render()

# Inicjalizacja okna renderowania VTK
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# --- source: read data
imageData = loadDICOMData()
if imageData is None:
    print("No DICOM series selected. Exiting...")
    exit()

nFrames = imageData.GetDimensions()[2]
winWidth = 750
winCenter = 100

# --- filter: apply winWidth and winCenter
shiftScaleFilter = vtk.vtkImageShiftScale()
shiftScaleFilter.SetOutputScalarTypeToUnsignedChar()            # output type
shiftScaleFilter.SetInputData(imageData)                       # input data
shiftScaleFilter.SetShift(-winCenter + 0.5 * winWidth)
shiftScaleFilter.SetScale(255.0 / winWidth)
shiftScaleFilter.SetClampOverflow(True)

# iso surface
isoValue = 128
contourFilter = vtk.vtkContourFilter()
contourFilter.SetInputConnection(shiftScaleFilter.GetOutputPort())
contourFilter.SetValue(0, isoValue)

# mapper for iso surface
isoMapper = vtk.vtkPolyDataMapper()
isoMapper.SetInputConnection(contourFilter.GetOutputPort())
isoMapper.ScalarVisibilityOff()

# actor for iso surface
isoActor = vtk.vtkActor()
isoActor.SetMapper(isoMapper)

# --- renderer
ren1 = vtk.vtkRenderer()
ren1.AddActor(isoActor)

# --- window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
renWin.SetSize(800, 600)

# --- interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# --- slider to change frame: callback class, sliderRepresentation, slider
class FrameCallback(object):
    def __init__(self, actor, renWin):
        self.renWin = renWin
        #self.actor = actor
        self.contourFilter = contourFilter
    def __call__(self, caller, ev):
        isoValue = caller.GetSliderRepresentation().GetValue()
        #actor.SetDisplayExtent(0, 255, 0, 255, int(value), int(value))
        self.contourFilter.SetValue(0, isoValue)
        self.renWin.Render()

sliderRep = vtk.vtkSliderRepresentation2D()
sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint1Coordinate().SetValue(.7, .1)
sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint2Coordinate().SetValue(.9, .1)
sliderRep.SetMinimumValue(0)
sliderRep.SetMaximumValue(nFrames - 1)
sliderRep.SetValue(nFrames // 2)
sliderRep.SetTitleText("iso")

slider = vtk.vtkSliderWidget()
slider.SetInteractor(iren)
slider.SetRepresentation(sliderRep)
slider.SetAnimationModeToAnimate()
slider.EnabledOn()
slider.AddObserver('InteractionEvent', FrameCallback(contourFilter, renWin))

# --- button to load DICOM series
buttonRep = vtk.vtkTexturedButtonRepresentation2D()
buttonRep.SetPlaceFactor(0.25)
buttonRep.PlaceWidget([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
#buttonRep.GetButtonProperty().SetFontSize(18)
#buttonRep.SetButtonTexture(vtk.vtkTexture(), 0)
# buttonRep.SetNumberOfStates(2)
# buttonRep.SetState(0)
# buttonRep.GetState(0).SetLabelText("Load DICOM Series")

buttonWidget = vtk.vtkButtonWidget()
buttonWidget.SetInteractor(iren)
buttonWidget.SetRepresentation(buttonRep)

def buttonCallback(obj, event):
    if buttonRep.GetState() == 0:
        newImageData = loadDICOMData()
        if newImageData is not None:
            updateData(newImageData)

buttonWidget.AddObserver("StateChangedEvent", buttonCallback)

# --- run
style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)
iren.Initialize()
iren.Start()
