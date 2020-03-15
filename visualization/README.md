# Visualization

## Attention Visualization

We will try to visualize the attention maps in our transformer based model.
First, install dependencies by first going to the `attention` folder.

```bash
cd attention
npm install
npm start
```

Then, you can go to `localhost:3000` in your webbrowser.

You'll also need to start up a server for the model.
You'll need to clone and install a separate repository, and you'll have to use
sudo because it overwrites part of allennlp.

```bash
git clone http://www.github.com/allenai/allennlp-server
cd allennlp-server
sudo pip install --editable . --user
```

Then, start up the model server.