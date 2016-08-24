local datasets = require 'datasets'
local paths = require 'paths'

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')


local DataProvider = torch.class('torch.DataProvider')

function DataProvider:__init(opt, dataset, split)
   self.pool = threads.Threads(
      opt.nThreads,
      function()
         require 'datasets'
      end,
      function(threadid)
         torch.manualSeed(opt.seed + threadid)
         threadDataset = dataset
      end
   )

   self.batchSize = opt.batchSize

   self.epochSize = dataset:size()
   self.count = 0
   if split == 'train' then
      self.split = split
      self.shuffle = opt.shuffle
      self.idcs = torch.randperm(self.epochSize)
   else
      self.split = 'test'
      self.idcs = torch.range(1, self.epochSize)
   end
end

local batch = nil
function DataProvider:getBatch()
   while self.pool:acceptsjob() do
      local start = self.count + 1
      self.count = self.count + self.batchSize

      local idcs
      if self.count <= self.epochSize then
         idcs = self.idcs[{{start, self.count}}]
      else
         if start <= self.epochSize then
            idcs = self.idcs[{{start, self.epochSize}}]
            self.count = self.batchSize - idcs:size(1)
         else
            self.count = self.batchSize
         end
         if self.shuffle then
            self.idcs = torch.randperm(self.epochSize)
         end
         if start <= self.epochSize then
            idcs = torch.cat(idcs, self.idcs[{{1, self.count}}], 1)
         else
            idcs = self.idcs[{{1, self.count}}]
         end
      end

      self.pool:addjob(
         function(idcs, split)
            return threadDataset:getBatch(idcs, split)
         end,
         function(_batch_)
            batch = _batch_
         end,
         idcs,
         self.split
      )
   end

   self.pool:dojob()
   return batch
end

function DataProvider:reset()
   self.pool:synchronize()
   self.count = 0
   if self.shuffle then
      self.idcs = torch.randperm(self.epochSize)
   end
end

local function siftflow(opt)
   local scene = torch.load(
      paths.concat(opt.dataDir, opt.dataset .. '.t7')
   )

   local mean = {128, 128, 128}

   local trainDataset = datasets.SceneDataset(
      scene.train.data, scene.train.labels, 3, 256, 256, mean, nil,
      true, {0.5, 1.5, 0.5, 1.5}
   )
   local trainDataProvider = torch.DataProvider(
      opt, trainDataset, 'train'
   )

   local testDataset = datasets.SceneDataset(
      scene.test.data, scene.test.labels, 3, 256, 256, mean, nil,
      nil, nil
   )
   local testDataProvider = torch.DataProvider(
      opt, testDataset, 'test'
   )

   return trainDataProvider, testDataProvider
end

local function getDataProvider(opt)
   if opt.dataset == 'siftflow' then
      return siftflow(opt)
   end
end

return getDataProvider
