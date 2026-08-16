//! Who may talk to the HTTP transports: a request carrying an `Origin`
//! is refused as a browser, and every request needs a bearer token —
//! loopback included, since that means a process, not a person.

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// The shared secret for one run of one server. Generated fresh per
/// start unless a caller supplies one; the `Debug` impl is redacted so
/// it appears only where it is handed to a human.
#[derive(Clone)]
pub struct AccessToken(String);

impl AccessToken {
    /// 24 bytes from the OS CSPRNG, base64url so it survives a shell, a
    /// JSON config file and a header value unquoted. `OsRng` rather than
    /// the thread RNG, so a secret reads as one at the call site.
    pub fn generate() -> Self {
        use base64::Engine;
        use rand::RngCore;

        let mut bytes = [0u8; 24];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        Self(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Adopt a token chosen by the caller.
    pub fn from_string(token: String) -> Self {
        Self(token)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Compare without leaking the answer in how long it took. The
    /// timing signal is buried under HTTP jitter in any realistic
    /// setting; the five lines cost nothing and remove the argument.
    fn matches(&self, presented: &str) -> bool {
        let expected = self.0.as_bytes();
        let presented = presented.as_bytes();
        if expected.len() != presented.len() {
            return false;
        }
        let mut diff = 0u8;
        for (a, b) in expected.iter().zip(presented) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

impl std::fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AccessToken(<redacted>)")
    }
}

/// Why a request never reached a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denial {
    /// The request carried an `Origin` — i.e. a browser sent it.
    BrowserOrigin,
    /// No `Authorization` header at all.
    MissingToken,
    /// An `Authorization` header that isn't a bearer token, or is the
    /// wrong one. One variant on purpose: telling a caller *which* of
    /// those it was tells an attacker whether the scheme is right.
    BadToken,
}

impl Denial {
    fn status(self) -> StatusCode {
        match self {
            Denial::BrowserOrigin => StatusCode::FORBIDDEN,
            Denial::MissingToken | Denial::BadToken => StatusCode::UNAUTHORIZED,
        }
    }

    /// Phrased for whoever has to fix it — an agent that can read the
    /// body, or a human reading a client's error log.
    fn message(self) -> &'static str {
        match self {
            Denial::BrowserOrigin => {
                "This MCP endpoint refuses requests carrying an Origin header: a browser \
                 is never a legitimate client, and accepting one would expose the open \
                 project to any page the user visits."
            }
            Denial::MissingToken => {
                "Missing bearer token. Send `Authorization: Bearer <token>`; the token is \
                 printed by the server at startup (Agent panel, or the log line for \
                 `voxelith mcp --http`)."
            }
            Denial::BadToken => {
                "Bearer token rejected. It changes every time the server restarts unless \
                 one was supplied with --token or VOXELITH_MCP_TOKEN."
            }
        }
    }
}

impl IntoResponse for Denial {
    fn into_response(self) -> Response {
        // A JSON-RPC error with no `id`, which the transport spec allows
        // for a request refused before parsing. A client reading only
        // the status still gets the right one.
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "error": { "code": -32001, "message": self.message() },
        });
        let mut response = (self.status(), axum::Json(body)).into_response();
        if self.status() == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

/// The whole policy, as a pure function over the request headers — so
/// it can be tested without a socket, and so the middleware below has
/// nothing in it worth reading twice.
pub fn check(headers: &HeaderMap, token: &AccessToken) -> Result<(), Denial> {
    if headers.contains_key(header::ORIGIN) {
        return Err(Denial::BrowserOrigin);
    }
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Err(Denial::MissingToken);
    };
    let Ok(value) = value.to_str() else {
        return Err(Denial::BadToken);
    };
    // RFC 9110: the auth scheme is case-insensitive, the credential is
    // not.
    let Some((scheme, presented)) = value.split_once(' ') else {
        return Err(Denial::BadToken);
    };
    if !scheme.eq_ignore_ascii_case("bearer") || !token.matches(presented.trim()) {
        return Err(Denial::BadToken);
    }
    Ok(())
}

/// Wrap a router so every request under it goes through [`check`].
pub fn guarded(router: axum::Router, token: AccessToken) -> axum::Router {
    router.layer(axum::middleware::from_fn_with_state(token, enforce))
}

async fn enforce(State(token): State<AccessToken>, request: Request, next: Next) -> Response {
    match check(request.headers(), &token) {
        Ok(()) => next.run(request).await,
        Err(denial) => {
            // At warn, not info: a rejected call is either a
            // misconfigured client the user is about to ask about, or
            // something the user should know reached their port.
            log::warn!("MCP HTTP request refused: {denial:?}");
            denial.into_response()
        }
    }
}

/// The line a human needs to point a client at a guarded server. Built
/// here rather than in the panel and the CLI separately — those are the
/// two places that would drift.
pub fn client_command(url: &str, token: &AccessToken) -> String {
    format!(
        "claude mcp add --transport http voxelith {url} --header \"Authorization: Bearer {}\"",
        token.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                header::HeaderName::from_bytes(name.as_bytes()).expect("test header name"),
                HeaderValue::from_str(value).expect("test header value"),
            );
        }
        map
    }

    fn token() -> AccessToken {
        AccessToken::from_string("s3cret-token".to_string())
    }

    #[test]
    fn the_right_token_gets_through() {
        let h = headers(&[("authorization", "Bearer s3cret-token")]);
        assert_eq!(check(&h, &token()), Ok(()));
    }

    #[test]
    fn the_scheme_is_case_insensitive_but_the_token_is_not() {
        let lower = headers(&[("authorization", "bearer s3cret-token")]);
        assert_eq!(check(&lower, &token()), Ok(()));

        let wrong_case = headers(&[("authorization", "Bearer S3CRET-TOKEN")]);
        assert_eq!(check(&wrong_case, &token()), Err(Denial::BadToken));
    }

    #[test]
    fn no_authorization_header_is_refused() {
        assert_eq!(
            check(&HeaderMap::new(), &token()),
            Err(Denial::MissingToken)
        );
    }

    #[test]
    fn a_wrong_or_malformed_credential_is_one_answer() {
        for value in [
            "Bearer wrong",
            "Bearer ",
            "Basic s3cret-token",
            "s3cret-token",
        ] {
            let h = headers(&[("authorization", value)]);
            assert_eq!(
                check(&h, &token()),
                Err(Denial::BadToken),
                "{value:?} should be refused as a bad token"
            );
        }
    }

    /// The rebinding guard outranks the credential: a browser that
    /// somehow *has* the token is still a browser.
    #[test]
    fn an_origin_header_is_refused_even_with_the_right_token() {
        let h = headers(&[
            ("origin", "https://evil.example"),
            ("authorization", "Bearer s3cret-token"),
        ]);
        assert_eq!(check(&h, &token()), Err(Denial::BrowserOrigin));
    }

    /// A same-length near-miss is the case the constant-time compare
    /// exists for, and the case a length check alone would let pass.
    #[test]
    fn a_token_differing_in_one_byte_is_refused() {
        let h = headers(&[("authorization", "Bearer s3cret-tokeN")]);
        assert_eq!(check(&h, &token()), Err(Denial::BadToken));
    }

    #[test]
    fn generated_tokens_are_unique_and_url_safe() {
        let a = AccessToken::generate();
        let b = AccessToken::generate();
        assert_ne!(a.as_str(), b.as_str());
        assert!(a.as_str().len() >= 32, "24 bytes of base64url");
        assert!(
            a.as_str()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{} should survive a shell and a header value unquoted",
            a.as_str()
        );
    }

    #[test]
    fn the_debug_impl_does_not_print_the_secret() {
        let printed = format!("{:?}", token());
        assert!(!printed.contains("s3cret"), "{printed}");
    }

    #[test]
    fn the_client_command_carries_the_header() {
        let line = client_command("http://127.0.0.1:8737/mcp", &token());
        assert!(
            line.contains("--header \"Authorization: Bearer s3cret-token\""),
            "{line}"
        );
    }
}
