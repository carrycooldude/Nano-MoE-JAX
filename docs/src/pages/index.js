import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          🧠 NanoMoE
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started →
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/quickstart"
            style={{ marginLeft: '1rem', color: 'white', borderColor: 'white' }}>
            Quickstart
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({ title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md" style={{ padding: '2rem 1rem' }}>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

const FeatureList = [
  {
    title: '🔀 Sparse MoE Routing',
    description: 'Top-K gating with softmax weighting activates only 2 of 4 experts per token — more capacity with constant compute.',
  },
  {
    title: '⚡ Pure JAX/Flax',
    description: 'Built from scratch with jax.vmap for parallel experts, jax.jit for XLA compilation, and jax.grad for automatic differentiation.',
  },
  {
    title: '📖 Educational & Hackable',
    description: '2.4M parameters, ~500 lines of code, fully tested. Perfect for learning MoE internals and experimenting with new ideas.',
  },
];

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="NanoMoE — Mixture-of-Experts in JAX"
      description="A lightweight, educational Mixture-of-Experts language model built from scratch in JAX/Flax">
      <HomepageHeader />
      <main>
        <section style={{ padding: '2rem 0' }}>
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
        <section style={{ padding: '2rem 0', textAlign: 'center' }}>
          <div className="container">
            <h2>Install in One Command</h2>
            <code style={{
              fontSize: '1.2rem',
              padding: '0.8rem 1.5rem',
              borderRadius: '8px',
              background: 'var(--ifm-color-emphasis-200)',
            }}>
              pip install nano-moe-jax
            </code>
          </div>
        </section>
      </main>
    </Layout>
  );
}
