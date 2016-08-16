
local cv = require 'cv'
require 'cv.objdetect'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'loadcaffe'



-- Viola-Jones face detector
local detectorParamsFile = './models/haarcascade_frontalface_default.xml'
local face_cascade = cv.CascadeClassifier{detectorParamsFile}

local fx = 0.5  -- rescale factor
local M = 227   -- input image size

-- Loading models
local gender_net = loadcaffe.load('./models/deploy_gender.prototxt', './models/gender_net.caffemodel'):float()
local img_mean = torch.load'./models/age_gender_mean.t7':permute(3,1,2):float()

local cap = cv.VideoCapture{0}
assert(cap:isOpened(), 'Failed to open ')

local ok, frame = cap:read{}

if not ok then
  print("Couldn't retrieve frame!")
  os.exit(-1)
end

while true do
  local start_time = os.clock()
  local w = frame:size(2)
  local h = frame:size(1)

  local im2 = cv.resize{frame, fx=fx, fy=fx}
  cv.cvtColor{im2, dst=im2, code=cv.COLOR_BGR2GRAY}

  local faces = face_cascade:detectMultiScale{im2}
  for i=1,faces.size do
    local f = faces.data[i]
    local x = f.x/fx
    local y = f.y/fx
    local w = f.width/fx
    local h = f.height/fx

      -- crop and prepare image for convnets
    local crop = cv.getRectSubPix{
      image=frame,
      patchSize={w, h},
      center={x + w/2, y + h/2},
    }

    if crop then
      local im = cv.resize{src=crop, dsize={256,256}}:float()
      local im2 = im - img_mean
      local I = cv.resize{src=im2, dsize={M,M}}:permute(3,1,2):clone()

      -- classify
      local gender_out = gender_net:forward(I)
      print(gender_out)
      local gender = gender_out[1] > gender_out[2] and 'M' or 'F'

      cv.rectangle{frame, pt1={x, y+3}, pt2={x + w, y + h}, color={30,255,30}}
      cv.putText{
        frame,
        gender,
        org={x, y},
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        color={255, 255, 0},
        thickness=1
      }
    end
  end

  print(string.format('time taken :%.2f',os.clock()-start_time))

  cv.imshow{"torch-OpenCV Age&Gender demo", frame}
  ok = cap:read{frame}

  if cv.waitKey{1} >= 0 or not ok then break end
end
