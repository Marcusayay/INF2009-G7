// src/pages/NotFoundPage.jsx
import { Link, useRouteError } from "react-router-dom";

export default function NotFoundPage() {
  const error = useRouteError();
  console.error(error); // Log the error for debugging

  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <h1>Oops! 😮</h1>
      <p>Sorry, an unexpected error has occurred.</p>
      <p>
        <i>{error.statusText || error.message}</i>
      </p>
      <Link to="/" style={{ marginTop: "1rem", display: "inline-block" }}>
        Go back to Safety
      </Link>
    </div>
  );
}