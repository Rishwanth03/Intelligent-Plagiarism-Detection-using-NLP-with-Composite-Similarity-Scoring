import { Suspense, lazy } from "react";

const UnifiedDashboard = lazy(() => import("./pages/UnifiedDashboard"));

function App() {
  return (
    <div className="app-shell">
      <Suspense fallback={<div className="panel" style={{ margin: "1rem" }}>Loading dashboard...</div>}>
        <UnifiedDashboard />
      </Suspense>
    </div>
  );
}

export default App;
