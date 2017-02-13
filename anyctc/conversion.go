package anyctc

import (
	"fmt"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

var internalCreator = anyvec64.DefaultCreator{}

// vectorTo64 creates a vector with []float64 numeric list
// types.
func vectorTo64(v anyvec.Vector) anyvec.Vector {
	switch d := v.Data().(type) {
	case []float64:
		return internalCreator.MakeVectorData(d)
	case []float32:
		s := make([]float64, len(d))
		for i, x := range d {
			s[i] = float64(x)
		}
		return internalCreator.MakeVectorData(s)
	default:
		panic(fmt.Sprintf("unsupported numeric type: %T", d))
	}
}

func batchesTo64(v []*anyseq.Batch) []*anyseq.Batch {
	res := make([]*anyseq.Batch, len(v))
	for i, x := range v {
		res[i] = &anyseq.Batch{
			Packed:  vectorTo64(x.Packed),
			Present: x.Present,
		}
	}
	return res
}

func batchesFrom64(c anyvec.Creator, v []*anyseq.Batch) []*anyseq.Batch {
	res := make([]*anyseq.Batch, len(v))
	for i, x := range v {
		slice := x.Packed.Data().([]float64)
		res[i] = &anyseq.Batch{
			Packed:  c.MakeVectorData(c.MakeNumericList(slice)),
			Present: x.Present,
		}
	}
	return res
}
