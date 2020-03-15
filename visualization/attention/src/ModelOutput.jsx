import React from 'react';
import HeatMap from './components/heatmap/HeatMap'
import Collapsible from 'react-collapsible'

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;

    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    var label = outputs['label'];
    var all_labels = outputs['all_labels'];
    var logits = outputs['logits'];
    var probs = outputs['probs'];
    // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // map component.
    // var attention_data = outputs['attention_data'].map(x => [x]);
    // This is a 2D attention matrix.
    //var matrix_attention_data = outputs['matrix_attention_data'];
    // Labels for our 2D attention matrix, and the rows in our 1D attention array.
    //var column_labels = outputs['column_labels'];
    //var row_labels = outputs['row_labels'];

    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    var xLabelWidth = "70px";

    return (
      <div className="model__content">

       {/*
         * TODO: This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}

        <div className="form__field">
          <label>Label</label>
          <div className="model__content__summary">{ label }</div>
          <div className="model__content__summary">{ String(logits.map((x) => x.toFixed(2))) }</div>
          <div className="model__content__summary">{ String(probs.map((x) => x.toFixed(2)))  }</div>
        </div>
      {/* 
        <div className="form__field">
          {}
          <Collapsible trigger="Model internals (beta)">
            <Collapsible trigger="1D attention">
                <HeatMap xLabels={['Column label']} yLabels={row_labels} data={attention_data} xLabelWidth={xLabelWidth} />
            </Collapsible>
            <Collapsible trigger="2D attention">
                <HeatMap xLabels={column_labels} yLabels={row_labels} data={matrix_attention_data} xLabelWidth={xLabelWidth} />
            </Collapsible>
          </Collapsible>
        </div> */}

        <div className="form__field">
            <Collapsible trigger="Model Information">
              <label>Possible Labels</label>
              <div>
                { all_labels.map((x, key) => <div key={key}> {x} </div>) }
              </div>
            </Collapsible>
        </div>
      </div>
    );

  }
}

export default ModelOutput;
