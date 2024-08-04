const path = require('path');

module.exports = {
  mode: 'development',
  entry: {
    background: './extension/background.ts',
    content: './extension/content.ts',
    popup: './extension/popup.ts'
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js'
  },
  resolve: {
    extensions: ['.ts', '.js']
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }
    ]
  },
  devtool: 'source-map', // Generate source maps for easier debugging
  optimization: {
    minimize: false // Disable code minification
  }
};
