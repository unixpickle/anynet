package anyconv

import (
	"image"
	"image/color"

	"github.com/unixpickle/anyvec"
)

// ImageToTensor converts an image to a tensor of RGB
// values.
// Values in the tensor range between 0 and 1.
func ImageToTensor(c anyvec.Creator, img image.Image) anyvec.Vector {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	minX := img.Bounds().Min.X
	minY := img.Bounds().Min.Y

	res := make([]float64, w*h*3)
	idx := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(minX+x, minY+y).RGBA()
			for _, comp := range []uint32{r, g, b} {
				res[idx] = float64(comp) / 0xffff
				idx++
			}
		}
	}

	return c.MakeVectorData(c.MakeNumericList(res))
}

// TensorToImage converts a tensor of RGB values into an
// image.
// Values in the tensor are clipped between 0 and 1.
//
// The anyvec.NumericList type must be []float32 or
// []float64.
func TensorToImage(width, height int, v anyvec.Vector) image.Image {
	var rawData []float64
	switch data := v.Data().(type) {
	case []float64:
		rawData = data
	case []float32:
		rawData = make([]float64, len(data))
		for i, x := range data {
			rawData[i] = float64(x)
		}
	}

	if len(rawData) != width*height*3 {
		panic("incorrect tensor size")
	}
	for i, x := range rawData {
		if x < 0 {
			rawData[i] = 0
		} else if x > 1 {
			rawData[i] = 1
		}
	}

	res := image.NewRGBA(image.Rect(0, 0, width, height))
	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var vals [3]uint8
			for z := 0; z < 3; z++ {
				vals[z] = uint8(rawData[idx]*0xff + 0.5)
				idx++
			}
			res.SetRGBA(x, y, color.RGBA{R: vals[0], G: vals[1], B: vals[2], A: 0xff})
		}
	}

	return res
}
