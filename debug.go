package anynet

import (
	"fmt"
	"io"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

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

func (d *Debug) println(args ...interface{}) {
	newArgs := append([]interface{}{"Debug (" + d.ID + "):"}, args...)
	if d.Writer == nil {
		fmt.Println(newArgs...)
	} else {
		fmt.Fprintln(d.Writer, newArgs...)
	}
}
