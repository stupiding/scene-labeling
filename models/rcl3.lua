local getRCL = require 'models/rcl'
local p = 0.5

local upsample = nn.SpatialUpSamplingNearest
local function getModel()
   local model = nn.Sequential()

   b1 = nn.Sequential()
   b1:add(cudnn.SpatialConvolution(3, 16, 5, 5, 1, 1, 2, 2))
   b1:add(nn.SpatialBatchNormalization(16))
   b1:add(cudnn.ReLU(true))

   b2 = nn.Sequential()
   b2:add(cudnn.SpatialConvolution(16, 32, 3, 3, 2, 2, 1, 1))
   b2:add(nn.SpatialBatchNormalization(32))
   b2:add(cudnn.ReLU(true))

   b3 = nn.Sequential()
   b3:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   b3:add(getRCL(32, 64, 3, 1, 1, 3, 3, false, 'pre'))

   b4 = nn.Sequential()
   b4:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   b4:add(nn.Dropout(p))
   b4:add(getRCL(64, 128, 3, 1, 1, 3, 3, false, 'pre'))

   b5 = nn.Sequential()
   b5:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   b5:add(nn.Dropout(p))
   b5:add(getRCL(128, 256, 3, 1, 1, 3, 3, false, 'pre'))

   b6 = nn.Sequential()
   b6:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   b6:add(nn.Dropout(p))
   b6:add(getRCL(256, 512, 3, 1, 1, 3, 3, false, 'pre'))

   model:add(b1)
   :add(nn.Concat(2)
      :add(nn.Sequential()
      :add(b2)
      :add(nn.Concat(2)
         :add(nn.Sequential()
         :add(b3)
         :add(nn.Concat(2)
            :add(nn.Sequential()
            :add(b4)
            :add(nn.Concat(2)
               :add(nn.Sequential()
               :add(b5)
               :add(nn.Concat(2)
                  :add(nn.Sequential()
                  :add(b6)
                  :add(upsample(32))
                  )
                  :add(upsample(16))
               ))
               :add(upsample(8))
            ))
            :add(upsample(4))
         ))
         :add(upsample(2))
      ))
      :add(nn.Identity())
   )
   :add(nn.Dropout(p))
   :add(cudnn.SpatialConvolution(1008, 33, 1, 1, 1, 1, 0, 0))

   model = nn.Sequential()
   :add(ConcatTable()
      :add(nn.Sequential():add(nn.SelectTable(1)):add(model))
      :add(nn.SelectTable(2))
   )
   :add(nn.MaskedSelect())

   model:cuda()
   model:get(1).gradInput = nil

   local criterion = nn.CrossEntropyCriterion():cuda()

   return model, criterion
end

return getModel
