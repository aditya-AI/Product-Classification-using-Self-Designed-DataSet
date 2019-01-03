count = 0
vidcap = cv2.VideoCapture('from_path')
success,image = vidcap.read()
success = True
while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("to_path/frame%d.jpg" % count, image)
  count = count + 1