import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';
import ErrorBoundary from './components/common/ErrorBoundary';
import ResearchPage from './pages/ResearchPage';
import ReportPage from './pages/ReportPage';
import HistoryPage from './pages/HistoryPage';
import ManualPage from './pages/ManualPage';
import ArchitecturePage from './pages/ArchitecturePage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen bg-[#faf7f0] text-[#1c1d22] font-sans selection:bg-[#ebe4d8] selection:text-[#1c1d22]">
        <Navbar />
        <main className="flex-1 pb-16">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<ResearchPage />} />
              <Route path="/report/:id" element={<ReportPage />} />
              <Route path="/research/:id" element={<ReportPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/manual" element={<ManualPage />} />
              <Route path="/architecture" element={<ArchitecturePage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
