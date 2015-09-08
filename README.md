# chainer-gogh

## Implementation of "A neural algorithm of Artistic style" (http://arxiv.org/abs/1508.06576)

- pip install chainer
- download network-in-network caffemodel from  https://gist.github.com/mavenlin/d802a5849de39225bcc6  (wget https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1 -O nin_imagenet.caffemodel)
- python chainer-gogh.py -i input.png -s style.png -o output.png -g 0

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/cat.png" height="150px">


<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_0.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im0.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_1.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im1.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_2.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im2.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_3.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im3.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_4.jpg" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im4.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_5.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im5.png" height="150px">

<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_6.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im6.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/style_7.png" height="150px">
<img src="https://raw.githubusercontent.com/mattya/chainer-gogh/master/sample_images/im7.png" height="150px">

## Usage:
# モデルをダウンロード
* NIN https://gist.github.com/mavenlin/d802a5849de39225bcc6
お手軽
* VGG https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
きれいな絵がかけるがとても重い。
VGGを使う際は、コード内のnin_forwardをvgg_forwardに書き換え、実行時に`-m VGG_ILSVRC_16_layers.caffemodel --width 256`を付ける。

### CPU実行
```
python chainer-gogh.py -i input.png -s style.png -o output.png -g -1
```

### GPU実行
```
python chainer-gogh.py -i input.png -s style.png -o output.png -g GPU番号
```

### VGG実行サンプル
```
python chainer-gogh.py -i input.png -s style.png -o output.png -g 0 -m VGG_ILSVRC_16_layers.caffemodel --width 256
```

### 複数枚同時生成
* まず、input.txtというファイル名で、以下の様なファイルを作る。
```
input0.png style0.png
input1.png style1.png
...
```
そして、chainer-gogh-multi.pyの方を実行
```
python chainer-gogh-multi.py -i input.txt
```
VGGを使うときはGPUのメモリ不足に注意

## パラメタについて
* `--lr`: 学習速度。生成の進捗が遅い時は大きめにする
* `--lam`: これを上げるとinput画像に近くなり、下げるとstyle画像に近くなる

## 注意
* 現在のところ画像は正方形に近いほうがいいです
