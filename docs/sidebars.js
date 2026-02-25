/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/expert-ffn',
        'architecture/router',
        'architecture/moe-layer',
        'architecture/attention',
        'architecture/transformer-block',
        'architecture/full-model',
      ],
    },
    {
      type: 'category',
      label: 'Training',
      items: [
        'training/pipeline',
        'training/load-balancing',
        'training/results',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/config',
        'api/layers',
        'api/model',
        'api/train',
        'api/utils',
      ],
    },
    'quickstart',
  ],
};

export default sidebars;
