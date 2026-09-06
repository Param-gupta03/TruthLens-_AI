import React from 'react';
import { AlertOctagon, RotateCcw, Home } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('TruthLens UI Error Boundary Caught Error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-[70vh] flex items-center justify-center p-6">
          <div className="max-w-lg w-full bg-white border border-[#ebe4d8] rounded-xl p-8 shadow-sm text-center">
            <div className="w-12 h-12 rounded-full bg-[#fef2f2] text-[#b91c1c] border border-[#fecaca] flex items-center justify-center mx-auto mb-4">
              <AlertOctagon size={24} />
            </div>

            <h2 className="font-serif text-xl font-bold text-[#1c1d22]">
              Interface Notice
            </h2>
            <p className="text-sm text-[#5d6069] mt-2 mb-5 leading-relaxed">
              An unexpected render exception occurred in this view. Backend APIs, model inference workers, and cached dossiers remain healthy.
            </p>

            {this.state.error?.message && (
              <div className="bg-[#f4efe6] border border-[#dfd5c6] rounded-md p-3 text-xs font-mono text-[#5d6069] mb-6 text-left overflow-x-auto">
                {this.state.error.message}
              </div>
            )}

            <div className="flex items-center justify-center gap-3">
              <button
                onClick={this.handleReload}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-white border border-[#dfd5c6] text-sm font-medium text-[#1c1d22] hover:bg-[#f4efe6] transition-colors"
              >
                <RotateCcw size={15} />
                <span>Reload Page</span>
              </button>
              <button
                onClick={this.handleReset}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-[#1c1d22] text-sm font-medium text-white hover:bg-[#2e3036] transition-colors shadow-sm"
              >
                <Home size={15} />
                <span>Return to Home</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
