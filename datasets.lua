local image = require 'image'
local datasets = {}

local function len(input)
   if type(input) == 'table' then
      return #input
   else
      return input:size(1)
   end
end


local SceneDataset = torch.class('SceneDataset', datasets)

function SceneDataset:__init(data, labels, c, h, w, mean, std,
   hFlip, resize)
   if labels then
      assert(len(data) == len(labels))
   end
   self.data = data
   self.labels = labels
   self.c = c or 3
   self.h = h or 256
   self.w = w or self.h

   self.mean = mean
   self.std = std
   self.hFlip = hFlip
   self.resize = resize
end

function SceneDataset:size()
   return len(self.data)
end

function SceneDataset:getBatch(idcs, split)
   local split = split or 'train'
   local h = self.h
   local w = self.w
   local b = idcs:size(1)

   local samples = {}
   local masks = {}
   local labels = {}
   for i = 1, b do
      local idx = idcs[i]
      local img = self.data[idx]
      local label = self.labels[idx]
      if torch.type(img) == 'string' then
         samples[i], masks[i], labels[i] = self:transform(img, label, split, h, w)
      else
         samples[i], masks[i], labels[i] = self:transform(img:clone(), label:clone(), split, h, w)
      end
   end
   local siz = samples[1]:size()
   samples = torch.FloatTensor.cat(samples, 1):view(b, siz[1], siz[2], siz[3])
   masks = torch.ByteTensor.cat(masks, 1):view(b, siz[1], siz[2], siz[3])
   labels = torch.cat(labels)

   local batch = {samples = samples, masks = masks, labels = labels}
   return batch
end

function SceneDataset:transform(img, label, split, h, w)
   if torch.type(img) == 'string' then
      img = image.load(img, self.c, 'byte')
      if label then
         label = image.load(label, 1, 'byte')
      end
   end

   if self.hFlip and torch.rand(1)[1] > 0.5 then
      img = image.hflip(img)
   end

   local mask = torch.ByteTensor(h, w):fill(0)
   local selected = nil
   if split == 'train' then
      selected = {torch.IntTensor(1):random(1, h)[1], torch.IntTensor(1):random(1, w)[1]}
      while label[selected[1]][selected[2]] > 33 do
         selected = {torch.IntTensor(1):random(1, h)[1], torch.IntTensor(1):random(1, w)[1]}
      end
      label = label[selected[1]][selected[2]]
   else
      local idcs = torch.le(label, 33)
      mask[idcs] = 1
      label = label[idcs]
   end

   if self.resize then
      local resize = self.resize
      if #resize == 2 then
         local r = resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1]
         img = image.scale(img, math.round(w * r), math.round(h * r))
         if selected then
            selected[1] = math.round(selected[1] * r)
            selected[2] = math.round(selected[2] * r)
         end
      else
         local rh = resize[1] + (resize[2] - resize[1]) * torch.rand(1)[1]
         loal rw = resize[3] + (resize[4] - resize[3]) * torch.rand(1)[1]
         img = image.scale(img, math.round(w * rw), math.round(h * rh))
         if selected then
            selected[1] = math.round(selected[1] * rh)
            selected[2] = math.round(selected[2] * rw)
         end
      end
   end

   img = img:float()
   if self.mean then
      if torch.type(self.mean) == 'number' then
         img = img - self.mean
      else
         for i = 1, #self.mean do
             img[i] = img[i] - self.mean[i]
         end
      end
   end
   if self.std then
      if torch.type(self.std) == 'number' then
         img = img / self.std
      else
         for i = 1, #self.std do
            img[i] = img[i] / self.std[i]
         end
      end
   end

   local sample
   if img:size(2) == h and img:size(3) == w then
      if selected then
         mask:fill(0)
         mask[selected[1]][selected[2]] = 1
      end
      sample = img
   elseif img:size(2) < h and img:size(3) < w then
      sample = torch.ByteTensor(3, h, w):fill(0)
      local hh = torch.IntTensor(1):random(0, h - img:size(2))[1]
      local ww = torch.IntTensor(1):random(0, w - img:size(3))[1]
      sample[{{}, {hh + 1, hh + img:size(2)}, {ww + 1, ww + img:size(3)}}] = img
   elseif img:size(2) < h and img:size(3) >= w then
      sample = torch.ByteTensor(3, h, w):fill(0)
      local hh = torch.IntTensor(1):random(0, h - img:size(2))[1]
      local ww = torch.IntTensor(1):random(0, img:size(3) - w)[1]
      sample[{{}, {hh + 1, hh + img:size(2)}, {}}] = img[{{}, {}, {ww + 1, ww + w}}]
   elseif img:size(2) >= h and img:size(3) < w then
      sample = torch.ByteTensor(3, h, w):fill(0)
      local hh = torch.IntTensor(1):random(0, img:size(2) - h)[1]
      local ww = torch.IntTensor(1):random(0, w - img:size(3))[1]
      sample[{{}, {}, {ww + 1, ww + img:size(3)}}] = img[{{}, {hh + 1, hh + h}, {}}]
   elseif img:size(2) >= h and img:size(3) >= w then
      sample = torch.ByteTensor(3, h, w):fill(0)
      local hh = torch.IntTensor(1):random(0, img:size(2) - h)[1]
      local ww = torch.IntTensor(1):random(0, img:size(3) - w)[1]
      sample = img[{{}, {hh + 1, hh + h}, {ww + 1, ww + w}}]
   end
      
   return sample, mask, label
end
