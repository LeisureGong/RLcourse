import shutil
from PIL import Image, ImageChops
def TrimImgEdge(inImgPath, outImgPath):
    r"""
    去除图片边框
    inImgPath: 输入图片路径
    outImgPath: 输出图片路径
    """
    print(f'TrimImgEdge {inImgPath} ...')
    imgIn = Image.open(inImgPath)
    # 创建一个边框颜色图片
    bg = Image.new(imgIn.mode, imgIn.size, imgIn.getpixel((0, 0)))
    diff = ImageChops.difference(imgIn, bg)
    # diff = ImageChops.add(diff, diff, 2.0, -10) # 可选，会去的更干净，副作用是误伤
    bbox = diff.getbbox()   # 返回左上角和右下角的坐标 (left, upper, right, lower)
    if bbox:
        imgIn.crop(bbox).save(outImgPath, quality=95)
    else:
        shutil.copyfile(inImgPath, outImgPath)

if __name__ == "__main__":
    TrimImgEdge('csharp.jpg', 'csharp_pillow.jpg')

