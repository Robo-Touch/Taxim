"""
this is the file to pre-process the raw data to the data pack
the data pack including: images, touch center, touch radius

Keyboard control:
left/right/up/down: control the circle's location
m/p: decrease/increase the circle's radius
f/c: decrease/increase the circle's moving step
"""

import gc
from glob import glob
from os import path as osp
import cv2
import numpy as np
import argparse

import nanogui as ng
from nanogui import Texture
from nanogui import glfw

# define the image size, should be consistent with the raw data size
w, h = 640, 480

parser = argparse.ArgumentParser()
parser.add_argument("-data_path", nargs='?', default='../data/calib_ball/',
                    help="Path to the collected raw tactile data.")
args = parser.parse_args()

class Circle:
  """the circle drawed on the tactile image to get the contact size"""
  color_circle = (0, 128, 0)
  radius = 75
  increments = 4
  opacity = 0.5
  def __init__(self, w, h):
    self.center = [h/2,w/2]

class CalibrateApp(ng.Screen):
  fnames = list()
  read_all = False
  load_img = True
  change = False
  bg_img_fn = None
  bg_id = None
  imgs = []
  touch_centers = []
  touch_radius = []
  names = []

  def __init__(self, data_path):
    super(CalibrateApp, self).__init__((1024, 768), "Gelsight Calibration App")

    window = ng.Window(self, "IO Window")
    window.set_position((15, 15))
    window.set_layout(ng.GroupLayout())

    ng.Label(window, "Folder dialog", "sans-bold")
    tools = ng.Widget(window)
    tools.set_layout(ng.BoxLayout(ng.Orientation.Horizontal,
                              ng.Alignment.Middle, 0, 6))
    b = ng.Button(tools, "Open")

    def cb():
        self.img_data_dir = ng.directory_dialog(data_path)
        print("Selected directory = %s" % self.img_data_dir)

        # check for background Frame
        # obtains fnames(currently accepts jpg/ppm/png)
        self.fnames = glob(osp.join(self.img_data_dir, "*.jpg")) +\
              glob(osp.join(self.img_data_dir, "*.ppm")) +\
              glob(osp.join(self.img_data_dir, "*.png")) +\
              glob(osp.join(self.img_data_dir, "*.jpeg"))
        self.fnames = sorted(self.fnames, key=lambda y: int(y.split("_")[-1].split(".")[0]))

        self.next_img_num = 0
        self.background_check(self.fnames)

    b.set_callback(cb)

    # image view
    self.img_window = ng.Window(self, "Current image")
    self.img_window.set_position((200, 15))
    self.img_window.set_layout(ng.GroupLayout())


    b = ng.Button(self.img_window, "Calibrate")
    # Calibrate button
    def calibrate_cb():
      list_to_name = self.fnames[self.next_img_num].split("/")
      frame = self.orig_img
      touch_center = self.circle.center
      radius = self.circle.radius
      print("the radius for frame {} is {}".format(self.next_img_num,radius))
      self.imgs.append(frame)
      self.touch_centers.append(touch_center)
      self.touch_radius.append(radius)
      self.names.append(list_to_name[-1])
      # Update img index
      self.load_img = True
      self.update_img_idx()


    b.set_callback(calibrate_cb)

    b = ng.Button(self.img_window, "Skip")
    # Calibrate button
    def skip_cb():
      print("Skip a frame")
      # Update img index
      self.load_img = True
      self.update_img_idx()


    b.set_callback(skip_cb)

    b = ng.Button(self.img_window, "Save Params")
    def cb():
      out_fn_path = osp.join(self.img_data_dir, "dataPack.npz")
      print("Saving params to %s"%out_fn_path)
      np.savez(out_fn_path,\
        f0 = self.bg_img,
        imgs = self.imgs,
        touch_center = self.touch_centers,
        touch_radius = self.touch_radius,
        names = self.names,
        img_size = self.bg_img.shape)
      print("Saved!")
    b.set_callback(cb)

    self.img_view = ng.ImageView(self.img_window)

    self.img_tex = ng.Texture(
                      pixel_format=Texture.PixelFormat.RGB,
                      component_format=Texture.ComponentFormat.UInt8,
                      size=[w, h],
                      min_interpolation_mode=Texture.InterpolationMode.Trilinear,
                      mag_interpolation_mode=Texture.InterpolationMode.Nearest,
                      flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
                  )

    self.perform_layout()

  def background_check(self, fnames):
    found = False
    for fnId, fn in enumerate(fnames):
      baseFn = osp.basename(fn)
      if(baseFn == "frame_0.ppm" or \
          baseFn == "frame_0.jpg" or \
          baseFn == "frame_0.png" or \
          baseFn == "ball0.jpeg"):
        self.bg_img_fn = fn
        self.bg_id = fnId

        self.bg_img = cv2.imread(self.bg_img_fn)
        found = True
        break

    if not found:
      print("No background Image Found! Looking for frame_0.ppm/frame0.jpg/frame0.png")
      self.set_visible(False)

  def update_img_idx(self):
    self.next_img_num += 1
    if(self.next_img_num == len(self.fnames)): self.read_all = True

  def overlay_circle(self,orig_img, circle):
    center = circle.center
    radius = circle.radius
    color_circle = circle.color_circle
    opacity = circle.opacity

    overlay = orig_img.copy()
    center_tuple = (int(center[0]), int(center[1]))
    cv2.circle(overlay, center_tuple, radius, color_circle, -1)
    cv2.addWeighted(overlay, opacity, orig_img, 1 - opacity, 0, overlay)
    return overlay

  def draw(self, ctx):
    # self.img_window.set_size((700,700))
    self.img_window.set_size((2000,2600))
    self.img_view.set_size((w, h))

    # load a new image
    if (self.load_img and len(self.fnames) > 0 and not self.read_all):
      if(self.next_img_num == self.bg_id): self.update_img_idx()
      print("Loading %s"%self.fnames[self.next_img_num])

      # Load img
      self.orig_img = cv2.imread(self.fnames[self.next_img_num])
      size = self.orig_img.shape
      self.circle = Circle(size[0], size[1])

    # Add circle and add img to viewer
    if((self.load_img and len(self.fnames) > 0) or self.change):
      self.load_img = False
      self.change = False
      # Add circle
      img = self.overlay_circle(self.orig_img, self.circle)

      if(self.img_tex.channels() > 3):
        height, width = img.shape[:2]
        alpha = 255*np.ones((height, width,1), dtype=img.dtype)
        img = np.concatenate((img, alpha), axis=2)

      # Add to img view
      self.img_tex.upload(img)
      self.img_view.set_image(self.img_tex)

    super(CalibrateApp, self).draw(ctx)

  def keyboard_event(self, key, scancode, action, modifiers):
    if super(CalibrateApp, self).keyboard_event(key, scancode,
                                          action, modifiers):
        return True
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        self.set_visible(False)
        return True
    elif key == glfw.KEY_C:
      self.circle.increments *= 2
    elif key == glfw.KEY_F:
      self.circle.increments /= 2
    else:
      self.change = True
      if(key == glfw.KEY_LEFT):
        self.circle.center[0] -= self.circle.increments
      elif(key == glfw.KEY_RIGHT):
        self.circle.center[0] += self.circle.increments
      elif(key == glfw.KEY_UP):
        self.circle.center[1] -= self.circle.increments
      elif(key == glfw.KEY_DOWN):
        self.circle.center[1] += self.circle.increments
      elif key == glfw.KEY_M:
        self.circle.radius -= 1
      elif key == glfw.KEY_P:
        self.circle.radius += 1

    return False


if __name__ == "__main__":
  ng.init()
  app = CalibrateApp(args.data_path)
  app.draw_all()
  app.set_visible(True)
  ng.mainloop(refresh=1 / 60.0 * 1000)
  del app
  gc.collect()
  ng.shutdown()
