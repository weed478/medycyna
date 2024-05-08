import vtk

reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName("mr_brainixA")
reader.Update()

volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputConnection(reader.GetOutputPort())

color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(0, 1.0, 1.0, 1.0)
color_transfer_function.AddRGBPoint(255, 1.0, 1.0, 1.0)

opacity_transfer_function = vtk.vtkPiecewiseFunction()
opacity_transfer_function.AddPoint(0, 0.0)

x = 100
h1 = 1
h2 = 3
opacity_transfer_function.AddPoint(x - h2, 0.0)
opacity_transfer_function.AddPoint(x - h1, 0.6)
opacity_transfer_function.AddPoint(x, 0.6)
opacity_transfer_function.AddPoint(x + h1, 0.6)
opacity_transfer_function.AddPoint(x + h2, 0.0)

opacity_transfer_function.AddPoint(255, 0.0)

volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_transfer_function)
volume_property.SetScalarOpacity(opacity_transfer_function)
volume_property.SetInterpolationTypeToLinear()

# volume
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0, 0.0, 0.0)
renderer.AddVolume(volume)

# window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Cube")
render_window.SetSize(800, 600)
render_window.AddRenderer(renderer)

# interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

class FrameCallback(object):
    def __init__(self, opacity, volume_property, ren_win, value_slider):
        self.ren_win = ren_win
        self.opacity = opacity
        self.volume_property = volume_property
        self.value_slider = value_slider

    def __call__(self, caller, ev):
        value = self.value_slider.GetValue()
        self.opacity.RemoveAllPoints()
        self.opacity.AddPoint(0, 0.0)
        x = value
        h1 = 1
        h2 = 10
        self.opacity.AddPoint(x - h2, 0.0)
        self.opacity.AddPoint(x - h1, 0.6)
        self.opacity.AddPoint(x, 0.6)
        self.opacity.AddPoint(x + h1, 0.6)
        self.opacity.AddPoint(x + h2, 0.0)
        self.opacity.AddPoint(255, 0.0)
        self.volume_property.SetScalarOpacity(self.opacity)
        self.ren_win.Render()

slider_rep = vtk.vtkSliderRepresentation2D()
slider_rep.SetMinimumValue(0)
slider_rep.SetMaximumValue(255)
slider_rep.SetValue(100)
slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
slider_rep.GetPoint1Coordinate().SetValue(0.5, 0.1)
slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
slider_rep.GetPoint2Coordinate().SetValue(0.9, 0.1)
slider_rep.SetTitleText("Value")

slider_widget = vtk.vtkSliderWidget()
slider_widget.SetInteractor(interactor)
slider_widget.SetRepresentation(slider_rep)
slider_widget.EnabledOn()

callback = FrameCallback(opacity_transfer_function, volume_property, render_window, slider_rep)
slider_widget.AddObserver("InteractionEvent", callback)

# run
interactor.Initialize()
render_window.Render()
interactor.Start()