-- @author : github.com/nitish11
-- @description : Detection of gender from given image file


local cv = require 'cv'
require 'cv.objdetect'
require 'cv.imgcodecs' -- reading/writing images
require 'cv.imgproc' -- image processing
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video input/output
require 'loadcaffe'  -- Caffe Model


local xtime = os.clock()
-- Viola-Jones face detector
local detectorParamsFile = './models/haarcascade_frontalface_default.xml'
local face_cascade = cv.CascadeClassifier{detectorParamsFile}

local fx = 0.5  -- rescale factor
local M = 227   -- input image size

-- Loading models
local gender_net = loadcaffe.load('./models/deploy_gender.prototxt', './models/gender_net.caffemodel'):float()
local img_mean = torch.load'./models/age_gender_mean.t7':permute(3,1,2):float()

local imagePath 
imagePath = arg[1]
loadType = cv.IMREAD_UNCHANGED
frame = cv.imread{imagePath, loadType}  
local w = frame:size(2)
local h = frame:size(1)
local area_threshold = 1000

local im2 = cv.resize{frame, fx=fx, fy=fx}
cv.cvtColor{im2, dst=im2, code=cv.COLOR_BGR2GRAY}

local faces = face_cascade:detectMultiScale{im2}

if faces.size > 0 then
  local f = faces.data[1]
  local x = f.x/fx
  local y = f.y/fx
  local w = f.width/fx
  local h = f.height/fx

  -- check area to deal with false face detections
  face_area = f.width*f.height

  if face_area > area_threshold then
    -- crop and prepare image for convnets
    local crop = cv.getRectSubPix{
      image=frame,
      patchSize={w, h},
      center={x + w/2, y + h/2},
    }

    if crop:nDimension() ~= 0 then
      local im = cv.resize{src=crop, dsize={256,256}}:float()
      local im2 = im - img_mean
      local I = cv.resize{src=im2, dsize={M,M}}:permute(3,1,2):clone()

      -- classify
      local gender_out = gender_net:forward(I)
      gender = gender_out[1] > gender_out[2] and 'Male' or 'Female'
    end
  end
end


if gender == 'Male' then
  print('Male')
elseif gender == 'Female' then
  print('Female')
else
  print('None')
end

