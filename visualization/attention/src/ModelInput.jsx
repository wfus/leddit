import React from 'react';
import Button from './components/Button'
import ModelIntro from './components/ModelIntro'


// TODO: These are some quickly-accessible examples to try out with your model.  They will get
// added to the select box on the demo page, and will auto-populate your input fields when they
// are selected.  The names here need to match what's read in `handleListChange` below.

const examples = [
  {
    short_text_input: "AITA for being a Navy Seal?",
    long_text_input: "What the fuck did you just fucking say about me, you little bitch? I'll have you know I graduated top of my class in the Navy Seals, and I've been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills. I am trained in gorilla warfare and I'm the top sniper in the entire US armed forces. You are nothing to me but just another target. I will wipe you the fuck out with precision the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to me over the Internet? Think again, fucker. As we speak I am contacting my secret network of spies across the USA and your IP is being traced right now so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your life. You're fucking dead, kid.",
  },
  {
    short_text_input: "AITA for telling my SO that she's chubby?",
    long_text_input: "For context, my SO is getting a bit chubby. She eats 6 meals a week, and her BMI is getting awfully close to 25. I'm worried for her health, but she refuses to go to the gym. AITA if I tell her she's fat to convince her to go to the local gym (Gold's Gym)?",
  },
  {
    short_text_input: "AITA for playing basketball outside of park hours?",
    long_text_input: "Playing basketball at the local school is against the rules after dusk, even though there are lamps and lights there. Am I the asshole for ballin outside of the official hours? The school districit never bothered to change the rules after the lights were installed.",
  },
];

// TODO: This determines what text shows up in the select box for each example.  The input to
// this function will be one of the items from the `examples` list above.
function summarizeExample(example) {
  return example.long_text_input.substring(0, 60);
}

const title = "AITA Model Visualization";
const description = (
  <span>
    Transformer based models for the AITA dataset.
  </span>
);

class ModelInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleListChange = this.handleListChange.bind(this);
    this.onClick = this.onClick.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      // TODO: This gets called when the select box gets changed.  You want to set the values of
      // your input boxes with the content in your examples.
      this.long_text_input.value = examples[e.target.value].long_text_input
      this.short_text_input.value = examples[e.target.value].short_text_input
    }
  }

  onClick() {
    const { runModel } = this.props;

    // TODO: You need to map the values in your input boxes to json values that get sent to your
    // predictor.  The keys in this dictionary need to match what your predictor is expecting to receive.
    runModel({selftext: this.long_text_input.value, title: this.short_text_input.value});
  }

  render() {

    const { outputState } = this.props;

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {examples.map((example, index) => {
                return (
                    <option value={index} key={index}>{summarizeExample(example) + "..."}</option>
                );
              })}
          </select>
        </div>

       {/*
         * TODO: This is where you add your input fields.  You shouldn't have to change any of the
         * code in render() above here.  We're giving a couple of example inputs here, one for a
         * larger piece of text, like a paragraph (the `textarea`) and one for a shorter piece of
         * text, like a question (the `input`).  You'll probably want to change the variable names
         * here to match the input variable names in your model.
         */}
        <div className="form__field">
          <label>Post Title</label>
          <input ref={(x) => this.short_text_input = x} type="text"/>
        </div>
        <div className="form__field">
          <label>Post Body</label>
          <textarea ref={(x) => this.long_text_input = x} type="text" autoFocus="true"></textarea>
        </div>

       {/* You also shouldn't have to change anything below here. */}

        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} onClick={this.onClick} />
        </div>
      </div>
    );
  }
}

export default ModelInput;
