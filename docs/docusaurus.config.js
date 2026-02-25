// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'NanoMoE',
  tagline: 'A lightweight Mixture-of-Experts language model in JAX/Flax',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://carrycooldude.github.io',
  baseUrl: '/Nano-MoE-JAX/',

  organizationName: 'carrycooldude',
  projectName: 'Nano-MoE-JAX',

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/carrycooldude/Nano-MoE-JAX/tree/main/docs/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl:
            'https://github.com/carrycooldude/Nano-MoE-JAX/tree/main/docs/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.jpg',
      colorMode: {
        defaultMode: 'dark',
        respectPrefersColorScheme: true,
      },
      mermaid: {
        theme: {light: 'neutral', dark: 'dark'},
      },
      navbar: {
        title: 'NanoMoE',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://pypi.org/project/nano-moe-jax/',
            label: 'PyPI',
            position: 'right',
          },
          {
            href: 'https://github.com/carrycooldude/Nano-MoE-JAX',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'Architecture',
                to: '/docs/architecture/overview',
              },
              {
                label: 'API Reference',
                to: '/docs/api/config',
              },
            ],
          },
          {
            title: 'Links',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/carrycooldude/Nano-MoE-JAX',
              },
              {
                label: 'PyPI',
                href: 'https://pypi.org/project/nano-moe-jax/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'Switch Transformer Paper',
                href: 'https://arxiv.org/abs/2101.03961',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} carrycooldude. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'toml'],
      },
    }),
};

export default config;
