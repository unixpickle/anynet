package anynet

import (
	"fmt"
	"io"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Debug{}).SerializerType(), DeserializeDebug)
}

// Debug is a layer which logs statistics about its
// inputs.
// Besides logging, the Debug layer does nothing to
// interfere with the flow of values in a network.
type Debug struct {
	// Writer to which stats are printed.
	// If nil, os.Stdout is used.
	Writer io.Writer

	ID            string
	PrintRaw      bool
	PrintMean     bool
	PrintVariance bool
}

// DeserializeDebug deserializes a Debug layer.
// The Writer will be nil.
func DeserializeDebug(d []byte) (*Debug, error) {
	var res Debug
	err := serializer.DeserializeAny(d, &res.ID, &res.PrintRaw, &res.PrintMean,
		&res.PrintVariance)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// Apply logs information about its input.
// The input is returned, untouched.
func (d *Debug) Apply(in anydiff.Res, n int) anydiff.Res {
	if d.PrintRaw {
		d.println("batch of", n, "values:", in.Output().Data())
	}
	cols := in.Output().Len() / n
	if d.PrintMean || d.PrintVariance {
		mean := anyvec.SumRows(in.Output(), cols)
		normalizer := mean.Creator().MakeNumeric(1 / float64(n))
		mean.Scale(normalizer)
		if d.PrintMean {
			d.println("mean:", mean.Data())
		}
		if d.PrintVariance {
			two := mean.Creator().MakeNumeric(2)
			squared := in.Output().Copy()
			anyvec.Pow(squared, two)
			variance := anyvec.SumRows(squared, cols)
			variance.Scale(normalizer)
			anyvec.Pow(mean, two)
			variance.Sub(mean)
			d.println("variance:", variance.Data())
		}
	}
	return in
}

// SerializerType returns the unique ID used to serialize
// a Debug layer with the serializer package.
func (d *Debug) SerializerType() string {
	return "github.com/unixpickle/anynet.Debug"
}

// Serialize serializes the layer.
func (d *Debug) Serialize() ([]byte, error) {
	return serializer.SerializeAny(d.ID, d.PrintRaw, d.PrintMean, d.PrintVariance)
}

func (d *Debug) println(args ...interface{}) {
	newArgs := append([]interface{}{"Debug (" + d.ID + "):"}, args...)
	if d.Writer == nil {
		fmt.Println(newArgs...)
	} else {
		fmt.Fprintln(d.Writer, newArgs...)
	}
}
