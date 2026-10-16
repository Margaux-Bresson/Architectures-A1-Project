import { useState } from 'react'
import type { FormEvent } from 'react'
import heroImg from './assets/hero.png'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import './App.css'

// URL du backend : définie au build via VITE_API_URL (voir le Dockerfile), localhost par défaut
const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type SearchResult = {
  line: string
  score: number
}

type SearchResponse = {
  query: string
  cluster: string
  results: SearchResult[]
}

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const text = query.trim()
    if (!text) return

    setLoading(true)
    setError(null)

    try {
      // Le contenu du champ texte est envoyé au endpoint de recherche du backend
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, k: 10 }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        const detail = body?.detail
        throw new Error(
          typeof detail === 'string'
            ? detail
            : `Erreur ${response.status} du serveur`,
        )
      }

      setData((await response.json()) as SearchResponse)
    } catch (err) {
      setData(null)
      setError(
        err instanceof TypeError
          ? `Impossible de joindre le backend (${API_URL})`
          : err instanceof Error
            ? err.message
            : 'Erreur inconnue',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <section id="center">
        <div className="hero">
          <img src={heroImg} className="base" width="170" height="179" alt="" />
          <img src={reactLogo} className="framework" alt="React logo" />
          <img src={viteLogo} className="vite" alt="Vite logo" />
        </div>
        <div>
          <h1>Get started</h1>
          <p>
            Edit <code>src/App.tsx</code> and save to test <code>HMR</code>
          </p>
        </div>
        <div className="search-box">
          <form className="search-form" onSubmit={handleSearch}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Votre recherche (ex. assurance véhicules)"
              aria-label="Texte de la recherche"
            />
            <button type="submit" disabled={loading || query.trim() === ''}>
              {loading ? 'Recherche…' : 'Search'}
            </button>
          </form>

          {error && <p className="search-error">{error}</p>}

          {data && (
            <section className="search-results">
              <p>
                Cluster le plus proche : <code>{data.cluster}</code>
              </p>
              {data.results.length === 0 ? (
                <p>Aucun résultat.</p>
              ) : (
                <ul>
                  {data.results.map((result, index) => (
                    <li key={`${index}-${result.line}`}>
                      <span className="result-line">{result.line}</span>
                      <span className="result-score">
                        {result.score.toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
        </div>
      </section>

      <div className="ticks"></div>

      <section id="next-steps">
        <div id="docs">
          <svg className="icon" role="presentation" aria-hidden="true">
            <use href="/icons.svg#documentation-icon"></use>
          </svg>
          <h2>Documentation</h2>
          <p>Your questions, answered</p>
          <ul>
            <li>
              <a href="https://vite.dev/" target="_blank">
                <img className="logo" src={viteLogo} alt="" />
                Explore Vite
              </a>
            </li>
            <li>
              <a href="https://react.dev/" target="_blank">
                <img className="button-icon" src={reactLogo} alt="" />
                Learn more
              </a>
            </li>
          </ul>
        </div>
        <div id="social">
          <svg className="icon" role="presentation" aria-hidden="true">
            <use href="/icons.svg#social-icon"></use>
          </svg>
          <h2>Connect with us</h2>
          <p>Join the Vite community</p>
          <ul>
            <li>
              <a href="https://github.com/vitejs/vite" target="_blank">
                <svg
                  className="button-icon"
                  role="presentation"
                  aria-hidden="true"
                >
                  <use href="/icons.svg#github-icon"></use>
                </svg>
                GitHub
              </a>
            </li>
            <li>
              <a href="https://chat.vite.dev/" target="_blank">
                <svg
                  className="button-icon"
                  role="presentation"
                  aria-hidden="true"
                >
                  <use href="/icons.svg#discord-icon"></use>
                </svg>
                Discord
              </a>
            </li>
            <li>
              <a href="https://x.com/vite_js" target="_blank">
                <svg
                  className="button-icon"
                  role="presentation"
                  aria-hidden="true"
                >
                  <use href="/icons.svg#x-icon"></use>
                </svg>
                X.com
              </a>
            </li>
            <li>
              <a href="https://bsky.app/profile/vite.dev" target="_blank">
                <svg
                  className="button-icon"
                  role="presentation"
                  aria-hidden="true"
                >
                  <use href="/icons.svg#bluesky-icon"></use>
                </svg>
                Bluesky
              </a>
            </li>
          </ul>
        </div>
      </section>

      <div className="ticks"></div>
      <section id="spacer"></section>
    </>
  )
}

export default App
