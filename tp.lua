require 'image'
-- require 'cutorch'
-- cutorch.setDevice(1)
-- img=torch.load("val_image_tabel_now.t7")
-- hot=torch.load("val_one_hot_table_now.t7")

-- print(#img)
-- print(#hot)
-- print(hot)
-- im=torch.CudaTensor(#img,3,224,224)
-- print(im:size())
-- ho=torch.Tensor(#hot)
-- print(ho:size())


-- for i=1,#img do
-- 	im[i]=img[i]
-- 	ho[i]=hot[i]
-- end
-- torch.save("val_images_now.t7",im)
-- torch.save("val_one_hot_now.t7",ho)
-- print("saved")
x="/media/navneet/74E8952AE894EC1C/INKERS/Interns/data/val_images/bedroom_val_lmdb/0/0/0/0/0/0/00000089629ce3ba87bae003073896ba01988dee.webp"
y=image.load(x)
print(#y)