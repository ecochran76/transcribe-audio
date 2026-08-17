const ICON_PATHS = {
  library: (
    <>
      <path d="M4 5.5h16v13H4z" />
      <path d="M8 5.5v13M12 5.5v13M16 5.5v13" />
      <path d="M3 9.5h18M3 14.5h18" />
    </>
  ),
  queue: (
    <>
      <path d="M9 6h11M9 12h11M9 18h11" />
      <path d="m3.5 6 1.25 1.25L7 4.75M3.5 12l1.25 1.25L7 10.75M3.5 18l1.25 1.25L7 16.75" />
    </>
  ),
  identity: (
    <>
      <circle cx="8.5" cy="8" r="3" />
      <path d="M3.5 19c.45-3.2 2.1-5 5-5 1.7 0 3 .62 3.86 1.8" />
      <circle cx="17" cy="15" r="3.5" />
      <path d="m19.6 17.6 2.4 2.4" />
    </>
  ),
  people: (
    <>
      <circle cx="9" cy="8" r="3" />
      <path d="M3.5 19c.5-3.35 2.33-5 5.5-5s5 1.65 5.5 5" />
      <path d="M15.5 5.25a3 3 0 0 1 0 5.5M16 14c2.65.25 4.15 1.92 4.5 5" />
    </>
  ),
  preview: (
    <>
      <path d="M2.75 12s3.25-5.25 9.25-5.25S21.25 12 21.25 12 18 17.25 12 17.25 2.75 12 2.75 12Z" />
      <circle cx="12" cy="12" r="2.75" />
    </>
  ),
  record: (
    <>
      <path d="M5 3.75h11l3 3V20.25H5z" />
      <path d="M8 3.75v5h7v-5M8 20.25v-7h8v7" />
      <path d="m10 16.5 1.35 1.35L14.5 14.7" />
    </>
  )
};

export function Icon({ name, size = 18 }) {
  return (
    <svg
      aria-hidden="true"
      className="ui-icon"
      fill="none"
      focusable="false"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7">
        {ICON_PATHS[name]}
      </g>
    </svg>
  );
}
