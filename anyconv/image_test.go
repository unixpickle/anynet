package anyconv

import (
	"image"
	"image/color"
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
)

func TestImageConversion(t *testing.T) {
	inImg := image.NewRGBA(image.Rect(1, 1, 3, 3))
	inImg.SetRGBA(1, 1, color.RGBA{R: 0x13, G: 0x37, B: 0x66, A: 0xff})
	inImg.SetRGBA(1, 2, color.RGBA{R: 0x37, G: 0x37, B: 0x66, A: 0xff})
	inImg.SetRGBA(2, 1, color.RGBA{R: 0x10, G: 0x37, B: 0x66, A: 0xff})
	inImg.SetRGBA(2, 2, color.RGBA{R: 0x5, G: 0x37, B: 0x66, A: 0xff})

	tensor := ImageToTensor(anyvec32.CurrentCreator(), inImg)
	outImg := TensorToImage(anyvec32.CurrentCreator(), 2, 2, tensor)

	for x := 0; x < 2; x++ {
		for y := 0; y < 2; y++ {
			oldR, oldG, oldB, _ := inImg.At(x+1, y+1).RGBA()
			newR, newG, newB, _ := outImg.At(x, y).RGBA()
			olds := []uint32{oldR, oldG, oldB}
			news := []uint32{newR, newG, newB}
			for z, expected := range olds {
				a := news[z]
				if expected/0x100 != a/0x100 {
					t.Errorf("value %d,%d,%d should be %d but got %d", x, y, z,
						expected/0x100, a/0x100)
				}
			}
		}
	}
}
